"""Write the start states of the Table 1 episodes to session directories.

``configs/mpc/table1/episodes.jsonl.gz`` holds every episode of the planning table: object,
episode name, goal fraction, planner seed per variant, task config and the stored simulator
state. The simulator reads these as files, so this script writes
``<out>/<object>/<episode>/{task_config.yaml,states/state_0.pkl}``, the layout that
``scripts/run_mpc_episode.sh`` expects. ``scripts/mpc_table1.py`` calls it by itself.

Usage: python scripts/table1_start_states.py [--out data/table1-start-states]
"""
import argparse
import gzip
import json
import os
import pickle
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE1_DIR = os.path.join(REPO, "configs", "mpc", "table1")
DEFAULT_OUT = os.path.join(REPO, "data", "table1-start-states")


def load_episodes():
    """All Table 1 episodes as dicts, sorted by object and episode name."""
    with gzip.open(os.path.join(TABLE1_DIR, "episodes.jsonl.gz"), "rt") as f:
        return [json.loads(line) for line in f]


def _write(path, data):
    """Write atomically, so parallel processes never read a partial file."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def write_session(episode, out):
    """Write one episode's session directory (if missing) and return its path."""
    session = os.path.join(out, episode["object"], episode["episode"])
    state_file = os.path.join(session, "states", "state_0.pkl")
    if not os.path.exists(state_file):
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        state = dict(episode["state"])
        for key in ("object_base_position", "object_base_orientation"):
            state[key] = {name: tuple(v) for name, v in state[key].items()}
        _write(os.path.join(session, "task_config.yaml"), episode["task_config"].encode())
        _write(state_file, pickle.dumps(state))
    return session


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    episodes = load_episodes()
    for e in episodes:
        write_session(e, a.out)
    print(f"{len(episodes)} start states in {a.out}")


if __name__ == "__main__":
    main()
