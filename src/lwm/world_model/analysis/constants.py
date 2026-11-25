"""Constants for LWM evaluation: colors, labels, object IDs, generalization categories."""

# ── Split colors and labels ──────────────────────────────────────────────────
# Standard colors: blue, orange, green, red
SPLIT_COLORS = {
    "expert": "#2196F3",    # blue
    "replay": "#FF9800",    # orange
    "new_config": "#4CAF50",  # green
    "total": "#757575",     # gray
}
SPLIT_LABELS = {
    "expert": "Expert",
    "replay": "Replay",
    "new_config": "Interaction",
    "total": "Total",
}

# ── Object metadata ──────────────────────────────────────────────────────────
SEEN_OBJECT_IDS = {"45526", "35059", "46092", "45166", "44853", "44817", "45092", "45132"}

OBJECT_CATEGORIES = {
    "door": {"ids": {"45443", "35059", "45526", "45249", "46092", "45166", "46417", "47388"}},
    "drawer": {"ids": {"44853", "44817", "45092", "45132", "45135", "45243", "45290", "45374"}},
}

# ── Generalization categories ────────────────────────────────────────────────
GEN_COLORS = {
    "seen_obj": "#2196F3",   # blue
    "unseen_obj": "#FF9800",  # orange
    "total": "#757575",       # gray
}
GEN_LABELS = {
    "seen_obj": "Seen Objects",
    "unseen_obj": "Unseen Objects",
    "total": "Total",
}
GEN_CATEGORIES = [
    "seen_obj",
    "unseen_obj",
]

# ── Phase-based evaluation ──────────────────────────────────────────────────
PHASE_NAMES = ["reach", "grasp", "pull"]
PHASE_LABELS = {"reach": "Reach", "grasp": "Grasp", "pull": "Pull"}
PHASE_COLORS = {"reach": "#2196F3", "grasp": "#FF9800", "pull": "#4CAF50"}

# ── Combined 4-way plot styles (source × generalization) ──────────────────
COMBINED_STYLES = {
    ("expert", "seen_obj"):       {"color": "#2196F3", "linestyle": "-",  "label": "Expert / Seen"},
    ("expert", "unseen_obj"):     {"color": "#2196F3", "linestyle": "--", "label": "Expert / Unseen"},
    ("new_config", "seen_obj"):   {"color": "#4CAF50", "linestyle": "-",  "label": "Interaction / Seen"},
    ("new_config", "unseen_obj"): {"color": "#4CAF50", "linestyle": "--", "label": "Interaction / Unseen"},
}
