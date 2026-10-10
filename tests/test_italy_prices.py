"""Unit tests for the pure helpers in data/italy_prices.py (no network)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from data.italy_prices import (
    daily_average_by_delivery_day,
    date_chunks,
    parse_energy_charts_price,
)


# ── date_chunks ──────────────────────────────────────────────────────────────

def test_date_chunks_cover_range_without_overlap():
    start, end = date(2025, 1, 1), date(2025, 12, 31)
    chunks = date_chunks(start, end, 90)
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    for (s0, e0), (s1, _e1) in zip(chunks, chunks[1:]):
        assert (s1 - e0).days == 1          # contiguous, no overlap
    for s, e in chunks:
        assert 1 <= (e - s).days + 1 <= 90  # inclusive length within the limit
    assert sum((e - s).days + 1 for s, e in chunks) == 365


def test_date_chunks_exact_multiple_and_single_day():
    assert date_chunks(date(2025, 1, 1), date(2025, 1, 10), 5) == [
        (date(2025, 1, 1), date(2025, 1, 5)),
        (date(2025, 1, 6), date(2025, 1, 10)),
    ]
    assert date_chunks(date(2025, 3, 3), date(2025, 3, 3), 90) == [(date(2025, 3, 3), date(2025, 3, 3))]


def test_date_chunks_empty_and_invalid():
    assert date_chunks(date(2025, 2, 1), date(2025, 1, 1), 30) == []
    with pytest.raises(ValueError):
        date_chunks(date(2025, 1, 1), date(2025, 1, 2), 0)


# ── parse_energy_charts_price ────────────────────────────────────────────────

def _payload(stamps, prices, unit="EUR / MWh"):
    return {"license_info": "CC BY 4.0", "unix_seconds": stamps, "price": prices, "unit": unit}


def test_parse_payload_builds_sorted_utc_series():
    stamps = [1790809200, 1790805600, 1790806500]   # deliberately unsorted
    s = parse_energy_charts_price(_payload(stamps, [30.0, 10.0, 20.0]))
    assert list(s.to_numpy()) == [10.0, 20.0, 30.0]
    assert str(s.index.tz) == "UTC"
    assert s.index[0] == pd.Timestamp("2026-09-30 22:00", tz="UTC")


def test_parse_payload_drops_missing_and_keeps_last_duplicate():
    stamps = [1790805600, 1790806500, 1790806500, 1790807400]
    s = parse_energy_charts_price(_payload(stamps, [10.0, 1.0, 2.0, None]))
    assert len(s) == 2
    assert s.iloc[-1] == 2.0


@pytest.mark.parametrize("payload", [
    None,
    "not a dict",
    {"unix_seconds": [1, 2], "price": [1.0]},          # length mismatch
    {"unix_seconds": [], "price": []},                  # empty
    _payload([1790805600], [12.0], unit="EUR / kWh"),   # wrong unit
])
def test_parse_payload_rejects_invalid_input(payload):
    assert parse_energy_charts_price(payload).empty


# ── daily_average_by_delivery_day ────────────────────────────────────────────

def _local_series(start_local: str, periods: int, freq: str, tz="Europe/Rome", values=None):
    idx = pd.date_range(start_local, periods=periods, freq=freq, tz=tz)
    vals = np.arange(periods, dtype=float) if values is None else np.asarray(values, dtype=float)
    return pd.Series(vals, index=idx).tz_convert("UTC")


def test_full_quarter_hour_day():
    s = _local_series("2026-10-05 00:00", 96, "15min")
    out = daily_average_by_delivery_day(s)
    assert len(out) == 1
    assert out.loc[0, "date"] == pd.Timestamp("2026-10-05")
    assert out.loc[0, "price"] == pytest.approx(np.arange(96).mean())
    assert out.loc[0, "n_points"] == 96
    assert out.loc[0, "coverage"] == pytest.approx(1.0)


def test_full_hourly_day():
    s = _local_series("2025-06-10 00:00", 24, "h", values=[50.0] * 24)
    out = daily_average_by_delivery_day(s)
    assert out.loc[0, "price"] == pytest.approx(50.0)
    assert out.loc[0, "coverage"] == pytest.approx(1.0)


def test_dst_short_and_long_days_are_complete():
    spring_hourly = _local_series("2026-03-29 00:00", 23, "h")       # 23-hour day
    spring_quarter = _local_series("2026-03-29 00:00", 92, "15min")  # 92 quarter-hours
    autumn_hourly = _local_series("2025-10-26 00:00", 25, "h")       # 25-hour day
    for s, expected_points in ((spring_hourly, 23), (spring_quarter, 92), (autumn_hourly, 25)):
        out = daily_average_by_delivery_day(s)
        assert len(out) == 1
        assert out.loc[0, "n_points"] == expected_points
        assert out.loc[0, "coverage"] == pytest.approx(1.0)


def test_partial_day_excluded_below_coverage_threshold():
    s = _local_series("2025-06-10 00:00", 12, "h")   # half a day
    assert daily_average_by_delivery_day(s, min_coverage=0.9).empty
    out = daily_average_by_delivery_day(s, min_coverage=0.4)
    assert out.loc[0, "coverage"] == pytest.approx(0.5)


def test_local_delivery_day_assignment_from_utc():
    # 22:00 UTC on 30 June is 00:00 CEST on 1 July: belongs to the 1 July delivery day.
    idx = pd.date_range("2026-06-30 22:00", periods=24, freq="h", tz="UTC")
    out = daily_average_by_delivery_day(pd.Series(1.0, index=idx))
    assert list(out["date"]) == [pd.Timestamp("2026-07-01")]


def test_mixed_resolution_across_days_and_single_point_day():
    hourly = _local_series("2025-09-30 00:00", 24, "h", values=[10.0] * 24)
    quarter = _local_series("2025-10-01 00:00", 96, "15min", values=[20.0] * 96)
    lone = _local_series("2025-10-02 00:00", 1, "h", values=[99.0])
    out = daily_average_by_delivery_day(pd.concat([hourly, quarter, lone]))
    assert list(out["date"]) == [pd.Timestamp("2025-09-30"), pd.Timestamp("2025-10-01")]
    assert list(out["price"]) == [pytest.approx(10.0), pytest.approx(20.0)]


def test_naive_index_rejected_and_empty_input():
    naive = pd.Series([1.0, 2.0], index=pd.date_range("2025-01-01", periods=2, freq="h"))
    with pytest.raises(ValueError):
        daily_average_by_delivery_day(naive)
    assert daily_average_by_delivery_day(pd.Series(dtype=float)).empty
