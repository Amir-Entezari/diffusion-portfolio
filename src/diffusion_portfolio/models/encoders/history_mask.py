"""Non-trainable model-space history masking for the controlled AR(1) ablation."""
from functools import partial

import torch


def history_mask_policy(cfg, keep_last):
    """Canonical policy; no fields are added to historical YAML/dataclass configs."""
    if keep_last is None:
        return None
    if (isinstance(keep_last, bool) or not isinstance(keep_last, int)
            or not 0 <= keep_last <= cfg.data.lookback):
        raise ValueError("history-keep-last must be an integer between 0 and lookback")
    if (cfg.data.dataset != "synthetic_mean_reverting" or cfg.data.lookback != 60
            or cfg.data.horizon != 1 or cfg.data.subtract_risk_free
            or cfg.model.history_encoder != "mlp" or cfg.model.type != "conditional_diffusion"):
        raise ValueError("History masking requires synthetic_mean_reverting, Vanilla MLP, lookback=60, horizon=1 and no RF subtraction")
    return {"keep_last": keep_last, "lookback": cfg.data.lookback,
            "convention": "zero earlier standardized observations (fitted training mean)"}


def mask_history(history, keep_last):
    """Mask only history, without modifying the supplied tensor or consuming RNG."""
    if (history.ndim != 3 or isinstance(keep_last, bool) or not isinstance(keep_last, int)
            or not 0 <= keep_last <= history.shape[1]):
        raise ValueError("Expected [batch, lookback, assets] and 0 <= keep_last <= lookback")
    if keep_last == history.shape[1]:
        return history
    masked = torch.zeros_like(history)
    if keep_last:
        masked[:, -keep_last:] = history[:, -keep_last:]
    return masked


def _mask_encoder_input(module, args, *, keep_last):
    return (mask_history(args[0], keep_last),)


def attach_history_mask(model, policy):
    """One pre-hook covers training, validation loss and scenario sampling.

    Hooks contain no parameters/buffers and do not change state_dict keys.
    Policy restoration/verification belongs to the saved training contract.
    """
    from diffusion_portfolio.models.encoders.mlp import HistoryEncoder

    if (not isinstance(model.history_encoder, HistoryEncoder)
            or model.lookback != policy["lookback"]):
        raise ValueError("History mask is incompatible with this encoder/lookback")
    model.history_encoder.register_forward_pre_hook(
        partial(_mask_encoder_input, keep_last=policy["keep_last"])
    )
