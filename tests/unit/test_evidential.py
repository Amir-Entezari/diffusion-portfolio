import torch

from diffusion_portfolio.models.uncertainty.evidential import (
    EvidentialRegimeHead,
    dirichlet_kl_to_uniform,
    evidential_classification_loss,
)


def test_evidential_head_shapes_and_probabilities():
    head = EvidentialRegimeHead(
        input_dim=8,
        n_classes=3,
    )

    features = torch.randn(
        5,
        8,
    )

    output = head(
        features
    )

    assert output.evidence.shape == (5, 3)
    assert output.alpha.shape == (5, 3)
    assert output.probabilities.shape == (5, 3)
    assert output.vacuity.shape == (5,)

    assert torch.all(
        output.evidence >= 0
    )

    assert torch.all(
        output.alpha >= 1
    )

    assert torch.allclose(
        output.probabilities.sum(
            dim=-1
        ),
        torch.ones(5),
        atol=1e-6,
    )

    assert torch.all(
        output.vacuity > 0
    )

    assert torch.all(
        output.vacuity <= 1
    )


def test_zero_evidence_approaches_total_vacuity():
    head = EvidentialRegimeHead(
        input_dim=4,
        n_classes=3,
    )

    with torch.no_grad():
        head.evidence_layer.weight.zero_()
        head.evidence_layer.bias.fill_(
            -100.0
        )

    output = head(
        torch.zeros(
            2,
            4,
        )
    )

    assert torch.allclose(
        output.vacuity,
        torch.ones(2),
        atol=1e-6,
    )


def test_uniform_dirichlet_has_zero_kl():
    alpha = torch.ones(
        4,
        3,
    )

    kl = dirichlet_kl_to_uniform(
        alpha
    )

    assert torch.allclose(
        kl,
        torch.zeros(4),
        atol=1e-6,
    )


def test_evidential_loss_is_finite_and_differentiable():
    head = EvidentialRegimeHead(
        input_dim=6,
        n_classes=3,
    )

    features = torch.randn(
        8,
        6,
    )

    targets = torch.tensor(
        [
            0,
            1,
            2,
            0,
            1,
            2,
            1,
            0,
        ]
    )

    output = head(
        features
    )

    loss_output = (
        evidential_classification_loss(
            output.alpha,
            targets,
            kl_weight=0.5,
        )
    )

    assert torch.isfinite(
        loss_output.loss
    )

    loss_output.loss.backward()

    assert (
        head.evidence_layer.weight.grad
        is not None
    )

    assert torch.isfinite(
        head.evidence_layer.weight.grad
    ).all()