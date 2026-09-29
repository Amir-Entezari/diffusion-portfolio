import numpy as np
import pytest

from diffusion_portfolio.data import (
    FACTOR_COLUMNS,
    parse_ff3_daily_text,
    parse_ff3_factor_text,
)


def fake_factor_text() -> str:
    return "\n".join(
        [
            "This file was created for testing.",
            "",
            ",Mkt-RF,SMB,HML,RF",
            "20200102,0.86,-0.97,-0.22,0.006",
            "20200103,-0.67,0.30,0.00,0.006",
            "",
        ]
    )


def test_rf_parser_reads_daily_values():
    rf = parse_ff3_daily_text(
        fake_factor_text()
    )

    assert len(rf.dates) == 2

    assert str(rf.dates[0].date()) == (
        "2020-01-02"
    )


def test_rf_percent_is_converted_to_decimal():
    rf = parse_ff3_daily_text(
        fake_factor_text()
    )

    # 0.006% = 0.00006
    assert rf.returns[0] == pytest.approx(
        0.00006
    )


def test_rf_parser_rejects_missing_sentinel():
    text = fake_factor_text().replace(
        "0.006",
        "-99.99",
        1,
    )

    with pytest.raises(
        ValueError,
        match="missing-value sentinel",
    ):
        parse_ff3_daily_text(text)


def test_rf_series_contains_finite_values():
    rf = parse_ff3_daily_text(
        fake_factor_text()
    )

    assert np.isfinite(
        rf.returns
    ).all()
    
    
def test_factor_parser_reads_all_four_factors():
    factors = parse_ff3_factor_text(
        fake_factor_text()
    )

    assert factors.returns.shape == (
        2,
        4,
    )

    assert factors.columns == (
        FACTOR_COLUMNS
    )

    np.testing.assert_allclose(
        factors.returns[0],
        np.array(
            [
                0.0086,
                -0.0097,
                -0.0022,
                0.00006,
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-8,
    )


def test_factor_rf_matches_legacy_rf_parser():
    factors = parse_ff3_factor_text(
        fake_factor_text()
    )

    rf = parse_ff3_daily_text(
        fake_factor_text()
    )

    rf_index = factors.columns.index(
        "RF"
    )

    np.testing.assert_allclose(
        factors.returns[
            :,
            rf_index,
        ],
        rf.returns,
    )