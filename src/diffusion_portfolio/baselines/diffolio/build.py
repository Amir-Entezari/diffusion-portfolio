"""Explicit construction and checkpoint restoration for the Diffolio baseline."""
from diffusion_portfolio.baselines.diffolio.objective import DiffolioObjective


def build_model(config, training_covariance):
    data, model = config["data"], config["model"]
    if model["prediction_type"] != "epsilon":
        raise ValueError("Diffolio reproduction requires epsilon prediction")
    if model["schedule"] != "linear":
        raise ValueError("Diffolio reproduction requires a linear schedule")
    return DiffolioObjective(
        training_covariance=training_covariance,
        n_assets=int(model["n_assets"]),
        n_asset_characteristics=int(model["n_asset_characteristics"]),
        n_systematic=int(model["n_systematic"]), lookback=int(data["lookback"]),
        hidden_dim=int(model["hidden_dim"]), num_heads=int(model["num_heads"]),
        mlp_dim=int(model["mlp_dim"]), time_embedding_dim=int(model["time_embedding_dim"]),
        diffusion_steps=int(model["diffusion_steps"]),
        beta_start=float(model["beta_start"]), beta_end=float(model["beta_end"]),
        lambda_corr=float(model["lambda_corr"]),
    )


def restore_model(config, checkpoint, device):
    state = checkpoint["model_state_dict"]
    model = build_model(config, state["training_covariance"].detach().clone())
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
