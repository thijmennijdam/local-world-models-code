"""Optional Weights & Biases import.

Training logs through ``wandb`` when it is installed. Logging is off unless
``WANDB_MODE`` is set (see ``lwm.world_model.training_utils.initialize_wandb``).
Without the package, a no-op stand-in with the subset of the API used here is
returned, so training runs without wandb installed.
"""

try:
    import wandb  # noqa: F401
except ImportError:  # pragma: no cover - depends on the environment
    import types

    def _noop(*args, **kwargs):
        return None

    wandb = types.SimpleNamespace(
        init=_noop,
        log=_noop,
        finish=_noop,
        summary={},
        Image=lambda data, *args, **kwargs: data,
    )
