"""Alpha-stable noise primitives for Phase 3."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _validate_alpha(
    alpha: float,
) -> float:
    alpha = float(
        alpha
    )

    if not (
        1.0
        < alpha
        <= 2.0
    ):
        raise ValueError(
            "alpha must satisfy 1 < alpha <= 2"
        )

    return alpha


def sample_positive_stable_mixer(
    *,
    alpha: float,
    shape: tuple[int, ...],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample the positive stable DLPM variance mixer.

    For diffusion tail index alpha in (1, 2), this samples

        A ~ S_{alpha/2, 1}(0, c_A)

    using the Chambers-Mallows-Stuck construction, with the
    normalization used by the official DLPM implementation.

    Under this normalization,

        sqrt(A) * G

    with G ~ N(0, I) is an isotropic alpha-stable vector
    with characteristic function

        E exp(i u^T X) = exp(-||u||^alpha).

    At alpha=2 the mixer becomes deterministically A=2,
    giving the Gaussian limit N(0, 2I).
    """

    alpha = _validate_alpha(
        alpha
    )

    if len(shape) == 0:
        raise ValueError(
            "shape cannot be empty"
        )

    if any(
        dimension <= 0
        for dimension in shape
    ):
        raise ValueError(
            "all shape dimensions must be positive"
        )

    if not dtype.is_floating_point:
        raise ValueError(
            "dtype must be floating point"
        )

    device = torch.device(
        "cpu"
        if device is None
        else device
    )

    if alpha == 2.0:
        return torch.full(
            shape,
            2.0,
            device=device,
            dtype=dtype,
        )

    stability = (
        alpha
        / 2.0
    )

    finfo = torch.finfo(
        dtype
    )

    uniform = torch.rand(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    # CMS requires an open interval.
    uniform = uniform.clamp(
        min=finfo.eps,
        max=1.0 - finfo.eps,
    )

    angle = (
        math.pi
        * (
            uniform
            - 0.5
        )
    )

    exponential = torch.empty(
        shape,
        device=device,
        dtype=dtype,
    )

    exponential.exponential_(
        1.0,
        generator=generator,
    )

    exponential = exponential.clamp_min(
        finfo.tiny
    )

    zeta = -math.tan(
        math.pi
        * stability
        / 2.0
    )

    xi = (
        math.atan(
            -zeta
        )
        / stability
    )

    shifted_angle = (
        stability
        * (
            angle
            + xi
        )
    )

    normalization = (
        1.0
        + zeta**2
    ) ** (
        1.0
        / (
            2.0
            * stability
        )
    )

    cosine_angle = torch.cos(
        angle
    ).clamp_min(
        finfo.tiny
    )

    cosine_term = torch.cos(
        angle
        - shifted_angle
    ).clamp_min(
        finfo.tiny
    )

    standard_positive = (
        normalization
        * torch.sin(
            shifted_angle
        )
        / torch.pow(
            cosine_angle,
            1.0
            / stability,
        )
        * torch.pow(
            cosine_term
            / exponential,
            (
                1.0
                - stability
            )
            / stability,
        )
    )

    # This factor matches the official DLPM implementation.
    #
    # The factor 2 is important for the stable-law convention:
    # alpha=2 corresponds to N(0, 2I), not N(0, I).
    scale = (
        2.0
        * math.cos(
            math.pi
            * alpha
            / 4.0
        ) ** (
            2.0
            / alpha
        )
    )

    mixer = (
        scale
        * standard_positive
    )

    if not torch.isfinite(
        mixer
    ).all():
        raise RuntimeError(
            "positive stable sampling produced "
            "non-finite values"
        )

    if torch.any(
        mixer <= 0
    ):
        raise RuntimeError(
            "positive stable mixer must be positive"
        )

    return mixer


def sample_isotropic_alpha_stable(
    *,
    alpha: float,
    shape: tuple[int, ...],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample an isotropic symmetric alpha-stable random vector.

    The final dimension is treated as the vector dimension.

    One positive stable scalar is shared across all coordinates
    of each vector:

        X = sqrt(A) G

    where

        G ~ N(0, I).

    Therefore the coordinates are conditionally independent
    given A, but marginally dependent, as required for the
    isotropic stable distribution used by DLPM.
    """

    alpha = _validate_alpha(
        alpha
    )

    if len(shape) == 0:
        raise ValueError(
            "shape cannot be empty"
        )

    if any(
        dimension <= 0
        for dimension in shape
    ):
        raise ValueError(
            "all shape dimensions must be positive"
        )

    device = torch.device(
        "cpu"
        if device is None
        else device
    )

    mixer_shape = (
        *shape[:-1],
        1,
    )

    mixer = (
        sample_positive_stable_mixer(
            alpha=alpha,
            shape=mixer_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    )

    gaussian = torch.randn(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    samples = (
        torch.sqrt(
            mixer
        )
        * gaussian
    )

    if not torch.isfinite(
        samples
    ).all():
        raise RuntimeError(
            "alpha-stable sampling produced "
            "non-finite values"
        )

    return samples