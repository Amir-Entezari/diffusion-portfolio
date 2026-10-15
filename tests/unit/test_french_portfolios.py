"""Offline fixtures for the intended French daily value-weighted tables."""
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.data.registry import load_return_table
from diffusion_portfolio.data.sources import kenneth_french
from diffusion_portfolio.data.sources.french_portfolios import (
    KF49_COLUMNS, KF100_SIZE_BM_COLUMNS, parse_french_daily_text,
    read_french_daily_zip,
)


FIXTURES = Path(__file__).parents[1] / "fixtures"
SOURCES = [
    ("ken_french_49", "ff49_daily.csv", KF49_COLUMNS),
    ("ken_french_100_size_bm", "ff100_size_bm_daily.csv", KF100_SIZE_BM_COLUMNS),
]


@pytest.fixture(params=SOURCES, ids=["ff49", "ff100"])
def source(request):
    dataset, filename, columns = request.param
    return dataset, (FIXTURES / filename).read_text(), columns


def write_archive(path, text):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("daily.csv", text)


def test_local_parser_and_dispatch_select_daily_vw_section(source, tmp_path, monkeypatch):
    dataset, text, columns = source
    monkeypatch.setattr(kenneth_french, "urlopen", lambda *a, **k: pytest.fail("Unexpected network request"))
    archive = tmp_path / "daily.zip"
    write_archive(archive, text)
    table = load_return_table(dataset, path=archive, download_if_missing=False)
    assert table.returns.shape == (3, len(columns))
    assert table.returns.dtype == np.float32
    assert table.columns == columns
    # Fixture headers are independent of the implementation's label constants.
    daily_header = text.split("Average Value Weighted Returns -- Daily\n")[1].splitlines()[0]
    assert columns == tuple(c.strip() for c in daily_header.split(",")[1:])
    assert table.dates.equals(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    assert table.returns[0, 0] == pytest.approx(0.0125)
    assert table.returns[0, -1] == pytest.approx(-0.005)
    np.testing.assert_array_equal(table.returns[1], np.full(len(columns), -.0025, np.float32))
    assert table.returns.max() < .02  # Not the equal-weighted/counts table.
    with pytest.raises(FileNotFoundError):
        load_return_table(dataset, path=tmp_path / "absent.zip", download_if_missing=False)


def test_official_label_counts_and_sort_order():
    assert len(KF49_COLUMNS) == len(set(KF49_COLUMNS)) == 49
    assert KF49_COLUMNS[:3] == ("Agric", "Food", "Soda")
    assert KF49_COLUMNS[-3:] == ("RlEst", "Fin", "Other")
    assert len(KF100_SIZE_BM_COLUMNS) == len(set(KF100_SIZE_BM_COLUMNS)) == 100
    assert KF100_SIZE_BM_COLUMNS[:2] == ("SMALL LoBM", "ME1 BM2")
    assert KF100_SIZE_BM_COLUMNS[9:12] == ("SMALL HiBM", "ME2 BM1", "ME2 BM2")
    assert KF100_SIZE_BM_COLUMNS[90] == "BIG LoBM"
    assert KF100_SIZE_BM_COLUMNS[-1] == "BIG HiBM"


@pytest.mark.parametrize("sentinel", ["-99.99", "-999"])
def test_missing_sentinels_fail_or_explicitly_drop_whole_dates(source, sentinel):
    _, text, columns = source
    text = text.replace("20200103,-0.25", "20200103," + sentinel)
    with pytest.raises(ValueError, match="sentinels in 1/3 daily rows"):
        parse_french_daily_text(text, columns=columns)
    with pytest.warns(UserWarning, match="2020-01-03.*retained 2 rows"):
        table = parse_french_daily_text(text, columns=columns, missing_policy="drop")
    assert table.returns.shape == (2, len(columns))
    assert pd.Timestamp("2020-01-03") not in table.dates
    assert table.returns[0, 0] == pytest.approx(.0125)
    np.testing.assert_array_equal(table.returns[1], np.full(len(columns), .005, np.float32))


def test_total_loss_is_not_a_missing_sentinel(source):
    _, text, columns = source
    text = text.replace("20200103,-0.25", "20200103,-100.00")
    table = parse_french_daily_text(text, columns=columns)
    assert table.returns.shape == (3, len(columns))
    assert table.returns[1, 0] == -1.0


@pytest.mark.parametrize("change, message", [
    (lambda t: t.replace("20200103", "20200102"), "chronological and unique"),
    (lambda t: t.replace("20200103", "20200101"), "chronological and unique"),
    (lambda t: t.replace("20200103", "20201303"), "Could not parse daily row"),
    (lambda t: t.replace("20200103", "202001"), "Malformed daily date"),
    (lambda t: t.replace("20200103,-0.25,", "20200103,"), "Malformed daily row"),
    (lambda t: t.replace("20200103,-0.25,", "20200103,1,2,"), "Malformed daily row"),
    (lambda t: t.replace("20200103,-0.25,", "20200103,,"), "Could not parse daily row"),
    (lambda t: t.replace("20200103,-0.25,", "20200103,nope,"), "Could not parse daily row"),
    (lambda t: t.replace("20200103,-0.25,", "20200103,nan,"), "NaN or infinite"),
    (lambda t: t.replace("20200103,-0.25,", "20200103,inf,"), "NaN or infinite"),
    (lambda t: t.replace("Average Value Weighted Returns -- Daily", "Average Value Weighted Returns -- Monthly"), "daily value-weighted"),
    (lambda t: t.replace("Average Value Weighted Returns -- Daily", "Average Equal Weighted Returns -- Daily"), "daily value-weighted"),
])
def test_malformed_data_is_never_silently_dropped(source, change, message):
    _, text, columns = source
    with pytest.raises(ValueError, match=message):
        parse_french_daily_text(change(text), columns=columns, missing_policy="drop")


def test_reordered_or_unexpected_columns_are_rejected(source):
    _, text, columns = source
    for new_columns in (columns[::-1], (*columns[:-1], columns[0]), (*columns, "extra")):
        with pytest.raises(ValueError, match="column"):
            parse_french_daily_text(text, columns=new_columns)
    with pytest.raises(ValueError, match="missing_policy"):
        parse_french_daily_text(text, columns=columns, missing_policy="zero_fill")


def test_no_complete_rows_is_an_error(source):
    _, text, columns = source
    for date, value in [("20200102", "1.25"), ("20200103", "-0.25"), ("20200106", "0.50")]:
        text = text.replace(date + "," + value, date + ",-999")
    with pytest.raises(ValueError, match="no complete daily rows"):
        parse_french_daily_text(text, columns=columns, missing_policy="drop")


def test_archive_validation(source, tmp_path):
    _, text, columns = source
    path = tmp_path / "bad.zip"
    path.write_text("not an archive")
    with pytest.raises(zipfile.BadZipFile):
        read_french_daily_zip(path, columns=columns)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.csv", text)
        archive.writestr("b.csv", text)
    with pytest.raises(ValueError, match="exactly one CSV"):
        read_french_daily_zip(path, columns=columns)


def test_shared_download_cache_is_atomic_and_preserves_kf12_default(tmp_path, monkeypatch):
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("daily.csv", "local fixture")
    requests = []
    def download(request, **kwargs):
        requests.append(request.full_url)
        return BytesIO(payload.getvalue())
    monkeypatch.setattr(kenneth_french, "urlopen", download)
    path = tmp_path / "data.zip"
    assert kenneth_french.download_kf12_daily(path) == path
    assert kenneth_french.download_kf12_daily(path) == path
    assert requests == [kenneth_french.KF12_DAILY_URL]
    original = path.read_bytes()
    monkeypatch.setattr(kenneth_french, "urlopen", lambda *a, **k: BytesIO(b"HTML error"))
    with pytest.raises(RuntimeError, match="valid ZIP"):
        kenneth_french.download_kf12_daily(path, overwrite=True)
    assert path.read_bytes() == original
    assert not path.with_suffix(".zip.tmp").exists()
