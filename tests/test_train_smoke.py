"""CPU training smoke runs for the experiment switches (E0, E2, E3, E5, E6, E7).

Each run trains a tiny LWM (d_model 64, 1 layer) for 10 epochs on a few episodes
(validation runs at epoch 10 and writes best.ckpt), then:
  - checks the switches in the run's .hydra/config.yaml and on the loaded model,
  - loads best.ckpt through the checkpoint loader,
  - runs one WorldModelPredictor.predict_trajectory call (MPC path; 12 steps = two
    chained 6-step calls) on a real observation,
  - runs the windowed eval (analysis.core.evaluate_windowed) on two windows,
  - checks that the rollout metrics on held-out windows (LWM_ROLLOUT_METRICS=1) and the
    throughput line were produced and that last.ckpt was written.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python tests/test_train_smoke.py [E0 E2 ...]
"""

import _common as C

import glob
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from omegaconf import OmegaConf

RUNS = {
    "E0": {},
    "E2": {"AFFORDANCE_TARGET_FRAME": "centroid", "INPUT_FRAME": "handle"},
    "E3": {"AFFORDANCE_TARGET_FRAME": "centroid", "INPUT_FRAME": "handle",
           "AUGMENT_YAW": "30", "AUGMENT_TRANSLATION": "0.05"},
    "E5": {"PREDICT_CONTACT": "gumbel_gate"},
    "E6": {"N_HEADS": "8"},
    "E7": {"PCD_TOKENS": "32", "PCD_GROUP_SIZE": "32"},
    # E3 trained from packed datasets (scripts/pack_dataset.py) instead of pickle dirs
    "E3_packed": {"AFFORDANCE_TARGET_FRAME": "centroid", "INPUT_FRAME": "handle",
                  "AUGMENT_YAW": "30", "AUGMENT_TRANSLATION": "0.05", "_packed": True},
}

SPLITS = {
    "expert": {"enabled": True, "step_budget": 600},
    "new_config": {"enabled": True, "step_budget": 600},
    "replay": {"enabled": False, "step_budget": 0},
}


def _train(name, switches, out_dir):
    env = dict(os.environ)
    env.pop("WANDB_MODE", None)
    # Validation uses DataLoader workers; a long TMPDIR breaks their socket paths (hangs).
    env.pop("TMPDIR", None)
    env.update({
        "PYTHONPATH": str(C.REPO / "src"),
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": out_dir,                       # norm-stat cache goes here
        "CHECKPOINT_BASE_DIR": out_dir,
        "EXP_NAME": f"smoke_{name}",
        "EXPERT_DATASET_PATH": C.EXPERT,
        "ROLLOUT_DATASET_PATH": C.INTERACTION,
        "DATA_SPLITS": json.dumps(SPLITS),
        "HORIZON": "6", "N_OBS_STEPS": "1", "BATCH_SIZE": "64", "NUM_EPOCHS": "10",
        "LEARNING_RATE": "1e-3", "LR_WARMUP_STEPS": "5", "WEIGHT_DECAY": "0.01",
        "D_MODEL": "64", "N_LAYERS": "1", "N_HEADS": "4",
        "PREDICT_AGENT_STATE": "true", "SKIP_ROTATION_NORMALIZATION": "false",
        "LWM_ROLLOUT_METRICS": "1", "LWM_ROLLOUT_WINDOWS": "32",
    })
    env.update({k: v for k, v in switches.items() if not k.startswith("_")})
    if switches.get("_packed"):
        from test_packed import _packed_dirs
        env["EXPERT_DATASET_PATH"], env["ROLLOUT_DATASET_PATH"] = _packed_dirs()
    cmd = [sys.executable, "-m", "lwm.world_model.train",
           "dataloader.num_workers=0", "val_dataloader.num_workers=0",
           "dataloader.persistent_workers=false", "val_dataloader.persistent_workers=false",
           "dataloader.pin_memory=false", "val_dataloader.pin_memory=false",
           "training.use_amp=false", "checkpoint.save_every=1000"]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(C.REPO), capture_output=True, text=True)
    log = os.path.join(out_dir, f"train_{name}.log")
    with open(log, "w") as f:
        f.write(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
        raise RuntimeError(f"training {name} failed (rc={proc.returncode}), log: {log}")
    ckpts = glob.glob(os.path.join(out_dir, "**", f"*_smoke_{name}", "checkpoints", "best.ckpt"), recursive=True)
    assert len(ckpts) == 1, ckpts
    assert os.path.exists(os.path.join(os.path.dirname(ckpts[0]), "last.ckpt")), "last.ckpt missing"
    out = proc.stdout + proc.stderr
    assert "Rollout metrics: rollout_seen_" in out, "rollout metrics were not computed"
    assert "samples/s" in out and "ETA" in out, "throughput line missing"
    roll = [l for l in out.splitlines() if "Rollout metrics:" in l][-1]
    print("   " + roll.strip()[:400])
    return ckpts[0], time.time() - t0


def _check_run(name, switches, ckpt):
    from lwm.world_model.analysis.checkpoint import load_checkpoint
    from lwm.planning.wm_predictor import WorldModelPredictor
    from lwm.world_model.analysis import core
    from lwm.world_model import evaluation_utils as eu

    run_dir = os.path.dirname(os.path.dirname(ckpt))
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    frame = switches.get("AFFORDANCE_TARGET_FRAME", "world")
    in_frame = switches.get("INPUT_FRAME", "world")
    assert cfg.dataset.affordance_target_frame == frame and cfg.model.affordance_target_frame == frame
    assert cfg.dataset.input_frame == in_frame and cfg.model.input_frame == in_frame
    assert float(cfg.dataset.augment_yaw) == float(switches.get("AUGMENT_YAW", 0))
    assert float(cfg.dataset.augment_translation) == float(switches.get("AUGMENT_TRANSLATION", 0))
    assert cfg.model.n_heads == int(switches.get("N_HEADS", 4))
    assert cfg.model.pcd_tokens == int(switches.get("PCD_TOKENS", 1))
    assert cfg.model.pcd_group_size == int(switches.get("PCD_GROUP_SIZE", 32))

    model, ckpt_dict, stats, _ = load_checkpoint(ckpt, device="cpu")
    assert model.affordance_target_frame == frame and model.input_frame == in_frame
    assert stats.get("affordance_target_frame", "world") == frame
    assert model.transformer.layers[0].self_attn.num_heads == int(switches.get("N_HEADS", 4))
    assert model.pcd_tokens == int(switches.get("PCD_TOKENS", 1))
    assert model.n_obs_tokens == model.pcd_tokens + 2
    if switches.get("PREDICT_CONTACT") == "gumbel_gate":
        assert model.predict_contact == "gumbel_gate"
        ident = torch.tensor([1., 0, 0, 0, 1, 0, 0, 0, 0])
        exp = (ident - torch.from_numpy(stats["affordance_delta_mean"])) / torch.from_numpy(stats["affordance_delta_std"])
        assert torch.allclose(model.identity_normalized, exp.float()), "identity_normalized not stored"

    ds = C.build_dataset(data_splits=SPLITS)
    rb = ds.replay_buffer
    ep = int(np.where(ds.val_mask)[0][0])
    states, affs, pcds, acts, L = eu.load_episode_data(rb, ep, "cpu")
    t, B = 0, 4
    actions = acts[t:t + 12].unsqueeze(0).repeat(B, 1, 1)
    wmp = WorldModelPredictor(ckpt, device="cpu")
    res = wmp.predict_trajectory(states[t].repeat(B, 1), affs[t].repeat(B, 1), pcds[t].repeat(B, 1, 1), actions)
    assert res["predicted_states"].shape == (B, 12, 10) and res["predicted_affordances"].shape == (B, 12, 3)
    assert torch.isfinite(res["predicted_states"]).all() and torch.isfinite(res["predicted_affordances"]).all()
    has_contact = "contact_logits" in res
    if switches.get("PREDICT_CONTACT") == "gumbel_gate":
        assert has_contact, "gumbel_gate model must return contact logits to the MPC"
    win = core.evaluate_windowed(model, rb, [ep], "cpu", stats, window_steps=6, windows_per_episode=2,
                                 compose_affordance_fn=eu.compose_affordance_with_9dof_delta, seed=0)
    assert len(win) > 0 and all(np.isfinite(w["aff_errors"]).all() for w in win)
    return {
        "best_epoch": ckpt_dict["epoch"],
        "val_loss": ckpt_dict["best_val_loss"],
        "mpc_aff_step12": float((res["predicted_affordances"][0, -1] - affs[t + 12]).norm() * 100),
        "contact_logits_in_mpc": has_contact,
        "windowed_step6_aff_cm": float(np.mean([w["aff_errors"][-1] for w in win])),
    }


def main(names):
    ok = True
    with tempfile.TemporaryDirectory(prefix="lwm_smoke_") as out_dir:
        for name in names:
            print(f"== smoke {name}: {RUNS[name] or 'defaults'}")
            try:
                ckpt, dt = _train(name, RUNS[name], out_dir)
                info = _check_run(name, RUNS[name], ckpt)
                print(f"   trained in {dt:.0f}s; {info}")
                print("   PASS")
            except Exception:
                import traceback
                traceback.print_exc()
                ok = False
                print("   FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:] or list(RUNS))
