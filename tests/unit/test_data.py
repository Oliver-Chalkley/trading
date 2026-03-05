import pandas as pd
import pytest

from trading.data import load_ohlcv

EXPECTED_COLUMNS = ["open", "high", "low", "close", "volume"]


def test_columns():
    df = load_ohlcv()
    assert list(df.columns) == EXPECTED_COLUMNS


def test_index_is_datetime():
    df = load_ohlcv()
    assert pd.api.types.is_datetime64_any_dtype(df.index)


def test_index_name():
    df = load_ohlcv()
    assert df.index.name == "timestamp"


def test_sorted_ascending():
    df = load_ohlcv()
    assert df.index.is_monotonic_increasing


def test_first_row():
    df = load_ohlcv()
    assert df.index[0] == pd.Timestamp("1993-01-29")


def test_row_count():
    df = load_ohlcv()
    assert len(df) > 8000  # noqa: PLR2004


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_ohlcv("/nonexistent/path/to/file.csv")


def test_custom_path(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2020-01-01,100.0,110.0,90.0,105.0,1000\n"
        "2020-01-02,105.0,115.0,95.0,110.0,2000\n"
    )
    df = load_ohlcv(csv_file)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 2  # noqa: PLR2004
    assert df.index[0] == pd.Timestamp("2020-01-01")
