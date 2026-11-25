"""Packed LWM datasets (scripts/pack_dataset.py): a few memory-mapped arrays instead of one
pickle per timestep.

A packed directory holds ``<field>.npy`` per per-step field (all episodes concatenated on
axis 0) and ``index.json`` with one entry per episode: ``key``, ``path`` (the episode
directory relative to the extracted root), ``start``/``end`` (rows in the arrays) and
``meta`` (the episode's constant, non-array fields of its first step, e.g.
goal_percentage, source_goal_percentage).

Episodes of a packed dataset are addressed by *virtual paths* ``<packed dir>/<path>``.
They look exactly like the pickle episode directories (same basename, same parent
``epoch-*`` dir), so object-id extraction, rollout classification and ``episode_key``
work unchanged. ``ReplayBuffer`` and ``LWMDataset`` resolve virtual paths through
:func:`lookup`.

The 4500-pt scene ``point_cloud`` is not packed (training and evaluation use
``part_point_cloud``); asking for it raises.
"""

import json
import os

import numpy as np

INDEX_FILE = "index.json"

_SOURCES = {}    # packed root (abspath) -> PackedSource
_EPISODES = {}   # virtual episode path -> (PackedSource, index entry)


def is_packed_dir(path):
    return path is not None and os.path.isfile(os.path.join(str(path), INDEX_FILE))


class PackedSource:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        with open(os.path.join(self.root, INDEX_FILE)) as f:
            idx = json.load(f)
        self.fields = idx["fields"]
        self.episodes = idx["episodes"]
        self._arrays = {}
        for e in self.episodes:
            _EPISODES[self.virtual_path(e)] = (self, e)

    def virtual_path(self, entry):
        return os.path.join(self.root, entry["path"])

    def array(self, key):
        if key not in self._arrays:
            if key not in self.fields:
                hint = (" The 4500-pt scene cloud is not packed; use part_point_cloud "
                        "(or repack with --keep-full-cloud)." if key == "point_cloud" else "")
                raise KeyError(f"field '{key}' is not in the packed dataset {self.root} "
                               f"(fields: {sorted(self.fields)}).{hint}")
            self._arrays[key] = np.load(os.path.join(self.root, f"{key}.npy"), mmap_mode="r")
        return self._arrays[key]


def open_packed(root):
    root = os.path.abspath(root)
    if root not in _SOURCES:
        _SOURCES[root] = PackedSource(root)
    return _SOURCES[root]


def lookup(path):
    """(PackedSource, entry) for a virtual episode path, or None for anything else.

    Opens the packed dataset on demand (walks up from ``path`` to the dir holding
    index.json), so it also works in processes that did not open it themselves.
    """
    path = os.path.abspath(os.fsdecode(path))
    hit = _EPISODES.get(path)
    if hit is not None:
        return hit
    if os.path.isdir(path):  # a real (pickle) episode directory
        return None
    d = os.path.dirname(path)
    while d and d != os.path.dirname(d):
        if is_packed_dir(d):
            open_packed(d)
            return _EPISODES.get(path)
        d = os.path.dirname(d)
    return None


def episode_paths(root, exclude_names=(), max_depth=5):
    """Virtual episode paths in the order the pickle loaders find the episode dirs.

    The pickle loaders walk the tree depth-first with children sorted by name, which is
    the lexicographic order of the path-component tuples. Episodes under excluded or
    hidden directories, or deeper than ``max_depth``, are dropped as there.
    """
    src = open_packed(root)
    keep = []
    for e in src.episodes:
        parts = e["path"].split("/")
        if len(parts) > max_depth:
            continue
        if any(p in exclude_names or p.startswith(".") for p in parts):
            continue
        keep.append((tuple(parts), src.virtual_path(e)))
    keep.sort(key=lambda x: x[0])
    return [p for _, p in keep]


def num_steps(path):
    hit = lookup(path)
    if hit is None:
        return None
    _, e = hit
    return e["end"] - e["start"]


def episode_meta(path):
    hit = lookup(path)
    return None if hit is None else hit[1]["meta"]


def has_field(path, key):
    hit = lookup(path)
    return hit is not None and key in hit[0].fields


def read_episode(path, keys, start=0, end=None):
    """Arrays [end-start, ...] of ``keys`` for one packed episode (copies, not views)."""
    src, e = lookup(path)
    n = e["end"] - e["start"]
    start = start + n if start < 0 else start
    end = n if end is None else (end + n if end < 0 else end)
    end = max(start, min(end, n))
    return {k: np.array(src.array(k)[e["start"] + start:e["start"] + end]) for k in keys}
