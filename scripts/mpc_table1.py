"""Run the paper's MPC episodes (Table 1) with a trained LWM, on the archived start states.

For every object, the first ``--n`` episodes (sorted by name) of the paper run are replayed: same
start state, goal and planner seed. Each episode runs ``lwm.planning.run_mpc`` with the planner
settings of the chosen variant; ``--workers`` episodes run in parallel on one GPU. Episodes whose
``metrics.json`` already exists are skipped, so a rerun with a larger ``--n`` extends the sample,
and a second process with ``--reverse`` on the same output directory works from the back.

Inputs (see docs/REPRODUCE.md):
  configs/mpc/table1/  the episodes (start state, goal, seeds) and the planner settings per variant;
              the start states are written to ``--sessions`` when they are missing
  --checkpoint the LWM checkpoint (``best.ckpt`` with ``.hydra/config.yaml`` next to it)
  PARTNET_DATASET_DIR  PartNet-Mobility assets (environment variable, required by run_mpc)

Variants: LWM-100M, LWM-100M_contact, LWM-100M_contact_kinematics and the same for LWM-200M
(no bias / contact bias / contact + kinematics bias).

Example (contact + kinematics row of LWM-200M, 15 episodes per object):
  python scripts/mpc_table1.py --variant LWM-200M_contact_kinematics \
      --checkpoint data/lwm-checkpoints/LWM-200M_d1024_L16/best.ckpt --n 15 --workers 6
Then: python scripts/table1_from_raw.py outputs/table1
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

from table1_start_states import DEFAULT_OUT, TABLE1_DIR, load_episodes, write_session

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPC_KEYS = ["planner_type", "horizon", "num_samples", "wm_batch_size", "num_iterations", "temperature",
            "elite_fraction", "alpha", "elites_keep_fraction", "noise_beta", "history_length",
            "history_elite_frac", "warmup_iters", "online_iters", "affordance_proximity_gate_scale",
            "finger_close_threshold", "use_gt_agent_state"]
COST_KEYS = ["cost_type", "ee_aff_weight", "aff_goal_weight", "terminal_weight", "pull_threshold",
             "clip_affordance", "grip_cost_weight", "proximity_scale", "progress_weight", "grip_enabled",
             "grip_hover_threshold", "grip_warmup_steps", "grip_close_value"]
EXEC_KEYS = ["num_replanning_steps", "goal_tolerance"]


def val(v):
    return "null" if v is None else str(v).lower() if isinstance(v, bool) else str(v)


def episodes(sessions, variant, n, objects=None):
    """(object, episode, seed, goal fraction, session dir) for the first n episodes per object."""
    out, count = [], {}
    for e in load_episodes():
        obj = e["object"]
        if variant not in e["seeds"] or (objects and obj not in objects):
            continue
        count[obj] = count.get(obj, 0) + 1
        if count[obj] <= n:
            out.append((obj, e["episode"], e["seeds"][variant], e["goal"], write_session(e, sessions)))
    return out


def overrides(variant, checkpoint, bounds, white, obj, seed, goal_pct, session, run_dir):
    d = yaml.safe_load(open(os.path.join(TABLE1_DIR, "planner.yaml")))[variant]
    ov = [f"checkpoint_path={checkpoint}", f"task.object_id='{obj}'", f"task.goal_percentage={goal_pct}",
          f"task.task_config_file={session}/task_config.yaml", f"task.init_state_file={session}/states/state_0.pkl",
          f"execution.seed={val(seed)}", f"logging.output_dir={run_dir}", f"hydra.run.dir={run_dir}",
          "logging.visualize=false", "logging.save_trajectories=true", "+logging.save_states=false"]
    ov += [f"mpc.{k}={val(0.0 if white and k == 'noise_beta' else d[k])}" for k in MPC_KEYS if k in d]
    ov += [f"cost.{k}={val(d[k])}" for k in COST_KEYS if k in d]
    ov += [f"execution.{k}={val(d[k])}" for k in EXEC_KEYS if k in d]
    ov += [f"task.action_bounds_source={bounds}"]
    return ov


def with_extra(ov, extra):
    """Apply --extra overrides, replacing any earlier override of the same key."""
    keys = {e.split("=", 1)[0] for e in extra}
    return [o for o in ov if o.split("=", 1)[0] not in keys] + list(extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="e.g. LWM-200M_contact_kinematics")
    ap.add_argument("--checkpoint", required=True, help="path to best.ckpt")
    ap.add_argument("--sessions", default=DEFAULT_OUT, help="where the start states are written")
    ap.add_argument("--n", type=int, default=15, help="episodes per object (the paper run has 50)")
    ap.add_argument("--workers", type=int, default=6, help="parallel episodes on one GPU (LWM-200M: ~4 GB each)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--objects", default=None, help="comma-separated subset of object ids")
    ap.add_argument("--out", default=os.path.join(os.environ.get("LWM_OUTPUT_DIR", "outputs"), "table1"))
    ap.add_argument("--tag", default="", help="suffix for the output directory (outputs/table1<tag>), e.g. _seed2")
    ap.add_argument("--bounds", choices=["checkpoint", "config"], default="checkpoint",
                    help="action bounds: the checkpoint's stored bounds (LWM-200M) or configs/mpc/planner.yaml "
                         "(what the LWM-100M rows use; its checkpoint stores none)")
    ap.add_argument("--white", action="store_true", help="white instead of colored action noise (noise_beta=0)")
    ap.add_argument("--extra", action="append", default=[], help="extra Hydra override, applied last")
    ap.add_argument("--reverse", action="store_true", help="process the episodes from the last object backwards")
    a = ap.parse_args()

    eps = episodes(a.sessions, a.variant, a.n, a.objects.split(",") if a.objects else None)
    if not eps:
        sys.exit(f"no episodes for variant {a.variant} (objects: {a.objects or 'all'})")
    if a.reverse:
        eps = eps[::-1]
    checkpoint = os.path.abspath(a.checkpoint)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    if "PARTNET_DATASET_DIR" not in env:
        sys.exit("set PARTNET_DATASET_DIR (PartNet-Mobility assets)")
    log_root = os.path.join(a.out + ("_white" if a.white else "") + a.tag, a.variant)

    def run(e):
        obj, ep, seed, goal_pct, ses = e
        run_dir = os.path.abspath(os.path.join(log_root, obj, ep))
        if os.path.exists(os.path.join(run_dir, "metrics.json")):
            return "skip", 0.0
        os.makedirs(run_dir, exist_ok=True)
        t = time.time()
        ov = overrides(a.variant, checkpoint, a.bounds, a.white, obj, seed, goal_pct,
                       os.path.abspath(ses), run_dir)
        with open(os.path.join(run_dir, "run.log"), "w") as log:
            r = subprocess.run([sys.executable, "-m", "lwm.planning.run_mpc", "--config-name", "planner"]
                               + with_extra(ov, a.extra), cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        ok = r.returncode == 0 and os.path.exists(os.path.join(run_dir, "metrics.json"))
        return ("ok" if ok else f"fail({r.returncode})"), time.time() - t

    print(f"{a.variant} [bounds={a.bounds}]: {len(eps)} episodes, {a.workers} workers, GPU {a.gpu} -> {log_root}",
          flush=True)
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for e, (status, dt) in zip(eps, ex.map(run, eps)):
            done += 1
            print(f"[{done}/{len(eps)}] {e[0]} {e[1]} {status} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
