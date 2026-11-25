#!/usr/bin/env python
"""Pack an extracted LWM dataset (one pickle per timestep) into a few memory-mappable files.

Input layout (HF lwm-expert-data / lwm-interaction-data, `extracted/`): episode directories that
contain 0.pkl, 1.pkl, ... Each pickle is a dict of per-step arrays with a leading dim of 1, plus
per-episode constants (goal percentage, goal source, joint metadata).

Output (in --out):
  <key>.npy     one array per per-step field, all episodes concatenated along axis 0 (float32;
                original_state_index int64)
  index.json    {"fields": {...}, "episodes": [{"key", "path", "start", "end", "meta"}]}
                key  = machine-independent episode id ("<epoch-dir>/<name>" or "<name>")
                path = episode directory relative to --src
                meta = the episode's non-array fields, read from step 0 (checked constant)

Usage:
  python scripts/pack_dataset.py --src <extracted dir> --out <packed dir> [--workers 16] [--keep-full-cloud]
  python scripts/pack_dataset.py --src <extracted dir> --out <packed dir> --verify 200
"""
import argparse
import json
import os
import pickle
import random
import sys
import time
from multiprocessing import Pool

import zipfile

import numpy as np

_ZIP = {}  # per-process open zip handle (zip mode)

STEP_KEYS = ["state", "action", "part_point_cloud", "static_affordance", "pcd_rigid_body_delta_9dof",
             "joint_angle", "joint_axis_base", "joint_axis_dir", "joint_type_id", "original_state_index"]
FULL_CLOUD_KEY = "point_cloud"
DROPPED = {"gripper_pcd", "goal_gripper_pcd", "displacement_gripper_to_object"}  # DP3-only fields


def episode_key(path):
    path = path.rstrip("/")
    parent, name = os.path.basename(os.path.dirname(path)), os.path.basename(path)
    return f"{parent}/{name}" if parent.startswith("epoch-") else name


def find_episodes(root):
    eps = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        if "0.pkl" in filenames:
            eps.append(dirpath)
            dirnames[:] = []  # episodes have no sub-episodes
        else:
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
    return sorted(eps)


def to_jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, dict):
        return {k: to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    return v


def find_episodes_zip(zip_path, prefix):
    """Episode dirs (zip member dirs under `prefix`) -> sorted step members."""
    eps = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.startswith(prefix) or not name.endswith(".pkl"):
                continue
            d, f = name.rsplit("/", 1)
            if f[:-4].isdigit():
                eps.setdefault(d, []).append(name)
    return {d: sorted(m, key=lambda n: int(n.rsplit("/", 1)[1][:-4])) for d, m in sorted(eps.items())}


def _load_steps(ep, zip_path):
    if zip_path is None:
        files = sorted((f for f in os.listdir(ep) if f.endswith(".pkl")), key=lambda f: int(f[:-4]))
        out = []
        for f in files:
            with open(os.path.join(ep, f), "rb") as fh:
                out.append(pickle.load(fh))
        return out
    if zip_path not in _ZIP:
        _ZIP[zip_path] = zipfile.ZipFile(zip_path)
    zf = _ZIP[zip_path]
    return [pickle.loads(zf.read(m)) for m in ep[1]]


def read_episode(args):
    ep, keys = args[0], args[1]
    zip_path = args[2] if len(args) > 2 else None
    ep_dir = ep if zip_path is None else ep[0]
    steps = _load_steps(ep, zip_path)
    first = steps[0]
    arrays = {}
    for k in keys:
        if k not in first:
            continue
        if k == "original_state_index":
            arrays[k] = np.array([int(s[k]) for s in steps], dtype=np.int64)
        else:
            arrays[k] = np.concatenate([np.asarray(s[k]).reshape(1, *np.asarray(s[k]).shape[1:]) for s in steps],
                                       axis=0).astype(np.float32)
    meta = {}
    for k, v in first.items():
        if k in keys or k in DROPPED or k == FULL_CLOUD_KEY:
            continue
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            continue
        meta[k] = to_jsonable(v)
    # per-episode constants must not change within the episode
    for s in steps[1:]:
        for k in meta:
            if to_jsonable(s.get(k)) != meta[k]:
                meta.setdefault("_inconsistent_meta", []).append(k)
                break
    return ep_dir, len(steps), arrays, meta


def pack(src, out, workers, keep_full_cloud, zip_path=None):
    """src: extracted dir, or (zip mode) the member prefix inside zip_path, e.g. 'extracted/'."""
    keys = STEP_KEYS + ([FULL_CLOUD_KEY] if keep_full_cloud else [])
    t0 = time.time()
    if zip_path is None:
        eps = find_episodes(src)
        jobs = [(e, keys) for e in eps]
    else:
        zeps = find_episodes_zip(zip_path, src)
        eps = list(zeps)
        jobs = [((d, m), keys, zip_path) for d, m in zeps.items()]
    print(f"found {len(eps)} episodes in {time.time() - t0:.0f}s", flush=True)
    os.makedirs(out, exist_ok=True)
    chunks = {k: [] for k in keys}
    index, start = [], 0
    present = None
    with Pool(workers) as pool:
        for i, (ep_dir, n, arrays, meta) in enumerate(pool.imap(read_episode, jobs, chunksize=4)):
            if present is None:
                present = [k for k in keys if k in arrays]
            missing = [k for k in present if k not in arrays]
            if missing:
                raise ValueError(f"{ep_dir} lacks {missing}; refusing to pack a ragged dataset")
            for k in present:
                assert arrays[k].shape[0] == n, (ep_dir, k)
                chunks[k].append(arrays[k])
            rel = os.path.relpath(ep_dir, src) if zip_path is None else ep_dir[len(src):].lstrip("/")
            index.append({"key": episode_key(ep_dir), "path": rel, "start": start, "end": start + n, "meta": meta})
            start += n
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(eps)} episodes, {start} steps, {time.time() - t0:.0f}s", flush=True)
    keys_seen = {}
    for e in index:
        keys_seen.setdefault(e["key"], 0)
        keys_seen[e["key"]] += 1
    dups = [k for k, c in keys_seen.items() if c > 1]
    if dups:
        raise ValueError(f"{len(dups)} duplicate episode keys, e.g. {dups[:3]}")
    fields = {}
    for k in present:
        arr = np.concatenate(chunks[k], axis=0)
        np.save(os.path.join(out, f"{k}.npy"), arr)
        fields[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        chunks[k] = None
    with open(os.path.join(out, "index.json"), "w") as fh:
        json.dump({"source": os.path.abspath(zip_path) + ":" + src if zip_path else os.path.abspath(src), "n_episodes": len(index), "n_steps": start,
                   "fields": fields, "episodes": index}, fh)
    bad = [e["path"] for e in index if "_inconsistent_meta" in e["meta"]]
    print(f"packed {len(index)} episodes / {start} steps in {time.time() - t0:.0f}s; "
          f"episodes with non-constant metadata: {len(bad)}", flush=True)


def verify(src, out, n, seed=0, zip_path=None):
    with open(os.path.join(out, "index.json")) as fh:
        idx = json.load(fh)
    arrays = {k: np.load(os.path.join(out, f"{k}.npy"), mmap_mode="r") for k in idx["fields"]}
    eps = random.Random(seed).sample(idx["episodes"], min(n, len(idx["episodes"])))
    worst = 0.0
    for e in eps:
        if zip_path is None:
            job = (os.path.join(src, e["path"]), list(idx["fields"]))
        else:
            d = src.rstrip("/") + "/" + e["path"]
            with zipfile.ZipFile(zip_path) as zf:
                members = sorted((n for n in zf.namelist() if n.startswith(d + "/") and n.endswith(".pkl")
                                  and n.rsplit("/", 1)[1][:-4].isdigit()), key=lambda n: int(n.rsplit("/", 1)[1][:-4]))
            job = ((d, members), list(idx["fields"]), zip_path)
        ep_dir, steps, ref, meta = read_episode(job)
        assert steps == e["end"] - e["start"], e["path"]
        assert meta == e["meta"], e["path"]
        for k, a in ref.items():
            packed = np.asarray(arrays[k][e["start"]:e["end"]])
            assert packed.shape == a.shape, (e["path"], k)
            worst = max(worst, float(np.max(np.abs(packed.astype(np.float64) - a.astype(np.float64)))) if a.size else 0.0)
    # every step of every episode is accounted for and contiguous
    ends = [(e["start"], e["end"]) for e in idx["episodes"]]
    assert ends[0][0] == 0 and all(a[1] == b[0] for a, b in zip(ends, ends[1:])) and ends[-1][1] == idx["n_steps"]
    print(f"verified {len(eps)} random episodes against the pickles: max abs diff {worst:.3g} "
          f"(float32 cast of the same values); index contiguous over {idx['n_steps']} steps")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--keep-full-cloud", action="store_true", help="also keep the 4500-pt scene cloud")
    ap.add_argument("--zip", default=None, help="read episodes from this zip; --src is then the member prefix (e.g. extracted/)")
    ap.add_argument("--verify", type=int, default=0, help="only verify N random episodes of an existing pack")
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify(a.src, a.out, a.verify, zip_path=a.zip) == 0.0 else 1)
    pack(a.src, a.out, a.workers, a.keep_full_cloud, zip_path=a.zip)
