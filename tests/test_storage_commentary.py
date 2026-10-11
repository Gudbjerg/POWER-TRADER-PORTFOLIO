"""Storage commentary: season-aware wording and the refill pace against the 5-year norm."""
from datetime import date

import pandas as pd
import pytest

from utils.scenarios import storage_refill_pace, storage_status


def _bands(start_fill: float = 50.0, per_day: float = 0.2) -> pd.DataFrame:
    """Synthetic 5-year mean path: rises linearly from 1 April and is flat outside the season."""
    rows = []
    for doy in range(1, 367):
        if 91 <= doy <= 305:   # 1 April to 1 November in a non-leap year
            mean = start_fill + (doy - 91) * per_day
        elif doy > 305:
            mean = start_fill + (305 - 91) * per_day
        else:
            mean = start_fill
        rows.append({"day_of_year": doy, "min": mean - 10, "mean": mean, "max": mean + 10})
    return pd.DataFrame(rows)


def test_pace_compares_required_injection_with_the_five_year_window():
    today = date(2026, 10, 10)   # day 283, 22 days before 1 November
    info = storage_refill_pace(73.5, _bands(per_day=0.2), today)
    assert info["days"] == 22
    assert info["required_pp"] == pytest.approx(16.5)
    assert info["norm_pp"] == pytest.approx(22 * 0.2)
    assert info["required_per_day"] == pytest.approx(16.5 / 22)
    assert info["verdict"] == "well above"


@pytest.mark.parametrize(
    ("pct", "verdict"),
    [(90.0 - 22 * 0.2, "in line with"), (90.0 - 22 * 0.2 * 1.2, "above"), (90.0 - 22 * 0.2 * 0.5, "below")],
)
def test_pace_verdict_thresholds(pct, verdict):
    info = storage_refill_pace(pct, _bands(per_day=0.2), date(2026, 10, 10))
    assert info["verdict"] == verdict


def test_target_already_met():
    info = storage_refill_pace(92.0, _bands(), date(2026, 9, 1))
    assert info["verdict"] == "met"
    assert info["required_pp"] == 0.0


def test_missing_bands_leave_the_norm_unknown():
    info = storage_refill_pace(70.0, pd.DataFrame(), date(2026, 10, 10))
    assert info["norm_pp"] is None and info["verdict"] is None
    assert info["required_pp"] == pytest.approx(20.0)


def test_after_november_the_next_target_date_is_used():
    info = storage_refill_pace(70.0, _bands(), date(2026, 12, 1))
    assert info["days"] == (date(2027, 11, 1) - date(2026, 12, 1)).days


def test_critical_wording_follows_the_season():
    _, _, autumn = storage_status(60.0, 70.0, 75.0, "EU", today=date(2026, 10, 10))
    _, _, winter = storage_status(30.0, 35.0, 45.0, "EU", today=date(2027, 1, 15))
    assert "before the winter withdrawal season" in autumn
    assert "summer" not in autumn
    assert "end of the withdrawal season" in winter
    for text in (autumn, winter):
        assert "90% by 1 November" in text
        assert "—" not in text


def test_lng_commentary_follows_the_season():
    from utils.scenarios import lng_status

    _, head_summer, detail_summer = lng_status(12.0, today=date(2026, 6, 1))
    _, head_winter, detail_winter = lng_status(12.0, today=date(2026, 12, 1))
    assert "refill season" in head_summer and "winter supply" in head_winter
    assert "injection" in detail_summer and "withdrawals" in detail_winter
    _, _, warn_winter = lng_status(-10.0, today=date(2027, 2, 1))
    assert "injection season" not in warn_winter


@pytest.mark.parametrize(
    ("value", "text"),
    [(0, "0th"), (1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"),
     (13, "13th"), (21, "21st"), (72.4, "72nd"), (99.6, "100th"), (float("nan"), "n/a"), (None, "n/a")],
)
def test_ordinal(value, text):
    from utils.helpers import ordinal

    assert ordinal(value) == text
