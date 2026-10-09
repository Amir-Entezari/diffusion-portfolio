"""Classification and calibration metrics for current regime probes."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def macro_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
) -> float:
    values = []

    for class_index in range(
        n_classes
    ):
        predicted_positive = (
            predictions
            == class_index
        )

        actual_positive = (
            targets
            == class_index
        )

        tp = (
            predicted_positive
            & actual_positive
        ).sum().item()

        fp = (
            predicted_positive
            & ~actual_positive
        ).sum().item()

        fn = (
            ~predicted_positive
            & actual_positive
        ).sum().item()

        precision = (
            tp
            / max(
                tp + fp,
                1,
            )
        )

        recall = (
            tp
            / max(
                tp + fn,
                1,
            )
        )

        if (
            precision
            + recall
            == 0
        ):
            values.append(
                0.0
            )

        else:
            values.append(
                2.0
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

    return float(
        np.mean(
            values
        )
    )



def expected_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_bins: int,
) -> float:
    confidence, predictions = (
        probabilities.max(
            dim=-1
        )
    )

    correct = (
        predictions
        == targets
    ).float()

    boundaries = torch.linspace(
        0.0,
        1.0,
        steps=n_bins + 1,
    )

    ece = 0.0

    for index in range(
        n_bins
    ):
        lower = boundaries[
            index
        ]

        upper = boundaries[
            index + 1
        ]

        if index == 0:
            mask = (
                confidence >= lower
            ) & (
                confidence <= upper
            )

        else:
            mask = (
                confidence > lower
            ) & (
                confidence <= upper
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        fraction = (
            count
            / len(targets)
        )

        bin_accuracy = (
            correct[
                mask
            ].mean().item()
        )

        bin_confidence = (
            confidence[
                mask
            ].mean().item()
        )

        ece += (
            fraction
            * abs(
                bin_accuracy
                - bin_confidence
            )
        )

    return float(
        ece
    )



def classification_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_classes: int,
    ece_bins: int,
    vacuity: torch.Tensor | None = None,
) -> dict:
    probabilities = (
        probabilities.detach().cpu()
    )

    targets = (
        targets.detach().cpu()
    )

    predictions = (
        probabilities.argmax(
            dim=-1
        )
    )

    accuracy = float(
        (
            predictions
            == targets
        )
        .float()
        .mean()
    )

    nll = float(
        F.nll_loss(
            torch.log(
                probabilities.clamp_min(
                    1e-8
                )
            ),
            targets,
        )
    )

    one_hot = F.one_hot(
        targets,
        num_classes=n_classes,
    ).float()

    brier = float(
        (
            (
                probabilities
                - one_hot
            )
            .square()
            .sum(
                dim=-1
            )
            .mean()
        )
    )

    result = {
        "accuracy": accuracy,
        "macro_f1": macro_f1(
            predictions,
            targets,
            n_classes,
        ),
        "nll": nll,
        "brier": brier,
        "ece": expected_calibration_error(
            probabilities,
            targets,
            n_bins=ece_bins,
        ),
    }

    if vacuity is not None:
        vacuity = (
            vacuity.detach().cpu()
        )

        correct = (
            predictions
            == targets
        )

        result[
            "mean_vacuity"
        ] = float(
            vacuity.mean()
        )

        if torch.any(
            correct
        ):
            result[
                "vacuity_correct"
            ] = float(
                vacuity[
                    correct
                ].mean()
            )

        else:
            result[
                "vacuity_correct"
            ] = None

        if torch.any(
            ~correct
        ):
            result[
                "vacuity_incorrect"
            ] = float(
                vacuity[
                    ~correct
                ].mean()
            )

        else:
            result[
                "vacuity_incorrect"
            ] = None

    return result
