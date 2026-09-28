from pathlib import Path
import zipfile

import numpy as np
import pytest

from diffusion_portfolio.data import (
    INDUSTRY_COLUMNS,
    parse_kf12_daily_text,
    read_kf12_daily_zip,
)


def fake_kf12_text() -> str:
    header = "," + ",".join(INDUSTRY_COLUMNS)

    return "\n".join(
        [
            "This file was created for testing.",
            "",
            "Average Value Weighted Returns -- Daily",
            header,
            "20200102,1.00,2.00,3.00,4.00,5.00,6.00,"
            "7.00,8.00,9.00,10.00,11.00,12.00",
            "20200103,-1.00,-2.00,-3.00,-4.00,-5.00,-6.00,"
            "-7.00,-8.00,-9.00,-10.00,-11.00,-12.00",
            "",
            "Average Equal Weighted Returns -- Daily",
            header,
            "20200102,99,99,99,99,99,99,99,99,99,99,99,99",
        ]
    )


def test_parser_selects_value_weighted_table():
    table = parse_kf12_daily_text(fake_kf12_text())

    assert table.returns.shape == (2, 12)

    # 1.00 percent -> 0.01 decimal return.
    assert table.returns[0, 0] == pytest.approx(0.01)

    # Make sure we did NOT accidentally parse the equal-weighted section.
    assert table.returns.max() < 1.0


def test_parser_preserves_dates():
    table = parse_kf12_daily_text(fake_kf12_text())

    assert str(table.dates[0].date()) == "2020-01-02"
    assert str(table.dates[1].date()) == "2020-01-03"


def test_parser_rejects_missing_value_sentinel():
    text = fake_kf12_text().replace(
        "20200103,-1.00",
        "20200103,-99.99",
    )

    with pytest.raises(
        ValueError,
        match="missing-value sentinel",
    ):
        parse_kf12_daily_text(text)


def test_zip_reader(tmp_path: Path):
    zip_path = tmp_path / "kf12.zip"

    with zipfile.ZipFile(
        zip_path,
        mode="w",
    ) as archive:
        archive.writestr(
            "12_Industry_Portfolios_Daily.csv",
            fake_kf12_text(),
        )

    table = read_kf12_daily_zip(zip_path)

    assert table.returns.shape == (2, 12)

    np.testing.assert_allclose(
        table.returns[0],
        np.arange(1, 13) / 100.0,
        rtol=1e-6,
    )