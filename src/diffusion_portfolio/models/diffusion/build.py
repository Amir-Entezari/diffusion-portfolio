"""Explicit config-to-model construction for current diffusion experiments."""
from diffusion_portfolio.models.diffusion.model import ConditionalDiffusionModel


def build_diffusion(cfg, *, n_assets: int, alpha=None, precomputed=False):
    model = cfg.model
    kwargs = dict(
        lookback=cfg.data.lookback,
        n_assets=n_assets,
        condition_dim=model.condition_dim,
        history_hidden_dim=model.history_hidden_dim,
        diffusion_steps=model.diffusion_steps,
        schedule_type=model.schedule,
        channels=list(model.channels),
        time_embed_dim=model.time_embed_dim,
        n_res_blocks=model.n_res_blocks,
        history_encoder_type="mlp" if precomputed else model.history_encoder,
    )
    # Cached-feature ablations historically construct an MLP before replacing
    # it with Identity. Preserve that initialization/RNG consumption exactly.
    if not precomputed:
        kwargs.update(
            cde_hidden_dim=model.cde_hidden_dim,
            cde_drift_hidden_dim=model.cde_drift_hidden_dim,
            cde_sensitivity_hidden_dim=model.cde_sensitivity_hidden_dim,
            cde_solver=model.cde_solver,
            cde_rtol=model.cde_rtol,
            cde_atol=model.cde_atol,
            cde_use_adjoint=model.cde_use_adjoint,
            cde_fixed_steps_per_interval=model.cde_fixed_steps_per_interval,
        )
    if alpha is not None:
        from diffusion_portfolio.models.diffusion.levy import ConditionalLevyDiffusionModel

        if model.prediction_type != "epsilon":
            raise ValueError("Lévy experiments require epsilon prediction")
        return ConditionalLevyDiffusionModel(alpha=alpha, **kwargs)
    return ConditionalDiffusionModel(prediction_type=model.prediction_type, **kwargs)
