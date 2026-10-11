import pytest

from diffusion_portfolio.baselines.diffolio import evaluation


def test_unknown_split_is_rejected_before_loading_data(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid split reached data acquisition and the held-out fallback")

    monkeypatch.setattr(evaluation, "prepare_diffolio_data", forbidden)
    with pytest.raises(ValueError, match="split"):
        evaluation.build_evaluation_dataset({"data": {}}, None, None, "validation")
