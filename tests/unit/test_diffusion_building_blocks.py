import torch

from diffusion_portfolio.models.diffusion import NoiseSchedule, ScoreNetwork


def test_noise_schedule_preserves_shape():
    schedule = NoiseSchedule(n_steps=20, schedule_type="cosine")
    x = torch.randn(4, 12)
    t = torch.tensor([0, 3, 7, 19])
    y = schedule.q_sample(x, t, torch.randn_like(x))
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_score_network_shape_and_gradient():
    net = ScoreNetwork(
        data_dim=12,
        channels=[16, 32],
        time_embed_dim=16,
        condition_dim=8,
        n_res_blocks=1,
    )
    x = torch.randn(4, 12, requires_grad=True)
    t = torch.tensor([1, 2, 3, 4])
    c = torch.randn(4, 8)
    y = net(x, t, c)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
