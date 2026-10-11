"""German coal generation from ENTSO-E A75 (data/generation.py): energy units and cache."""
import pandas as pd
import pytest

from data import generation as gen

HARD_COAL_MW = 4_000.0
LIGNITE_MW = 10_000.0


def _power(start: str, periods: int, freq: str = "15min") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame({"hard_coal_mw": HARD_COAL_MW, "lignite_mw": LIGNITE_MW}, index=index)


def _entsoe_frame(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """What EntsoePandasClient.query_generation returns for Germany: 15-minute MW readings."""
    index = pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC")
    columns = pd.MultiIndex.from_tuples([
        ("Fossil Hard coal", "Actual Aggregated"),
        ("Fossil Brown coal/Lignite", "Actual Aggregated"),
        ("Wind Onshore", "Actual Aggregated"),
    ])
    frame = pd.DataFrame(index=index, columns=columns, dtype=float)
    frame[("Fossil Hard coal", "Actual Aggregated")] = HARD_COAL_MW
    frame[("Fossil Brown coal/Lignite", "Actual Aggregated")] = LIGNITE_MW
    frame[("Wind Onshore", "Actual Aggregated")] = 20_000.0
    return frame


class _FakeClient:
    def __init__(self):
        self.calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def query_generation(self, country_code, start, end, psr_type=None):
        self.calls.append((start, end))
        return _entsoe_frame(start, end)


# ── Energy from power readings ────────────────────────────────────────────────

def test_fifteen_minute_readings_are_not_counted_four_times():
    daily = gen._daily_energy_gwh(_power("2026-03-02", 2 * 96))
    assert list(daily.columns) == gen._CACHE_COLUMNS
    assert len(daily) == 2
    # 4 GW for 24 hours is 96 GWh; 10 GW for 24 hours is 240 GWh.
    assert daily["hard_coal_gwh"].tolist() == [96.0, 96.0]
    assert daily["lignite_gwh"].tolist() == [240.0, 240.0]


def test_energy_does_not_depend_on_the_reporting_resolution():
    quarter_hourly = gen._daily_energy_gwh(_power("2026-03-02", 96))
    hourly = gen._daily_energy_gwh(_power("2026-03-02", 24, freq="h"))
    pd.testing.assert_frame_equal(quarter_hourly, hourly)


def test_days_still_being_published_are_left_out():
    # Two full days, then ten hours of the third.
    daily = gen._daily_energy_gwh(_power("2026-03-02", 2 * 96 + 40))
    assert daily["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-03-02", "2026-03-03"]


def test_one_missing_hour_is_tolerated():
    power = _power("2026-03-02", 96)
    power.iloc[40:44] = float("nan")   # 10:00 to 11:00 UTC not published
    daily = gen._daily_energy_gwh(power)
    assert len(daily) == 1
    assert daily["hard_coal_gwh"].iloc[0] == pytest.approx(92.0)


def test_days_are_utc_and_dates_are_naive():
    power = _power("2026-03-02", 96).tz_convert("Europe/Berlin")
    daily = gen._daily_energy_gwh(power)
    assert daily["date"].dt.tz is None
    assert daily["date"].iloc[0] == pd.Timestamp("2026-03-02")


# ── Cache files ───────────────────────────────────────────────────────────────

def test_version_one_cache_is_not_read(tmp_path, monkeypatch):
    """The v1 file summed 15-minute readings as hourly energy (four times too high)."""
    (tmp_path / "generation_cache.csv").write_text(
        "date,hard_coal_gwh,lignite_gwh\n2026-03-02,384.0,960.0\n", encoding="utf-8")
    monkeypatch.setattr(gen, "_CACHE_CSV", str(tmp_path / gen._CACHE_NAME))
    monkeypatch.setattr(gen, "_SEED_CSV", str(tmp_path / "missing" / gen._CACHE_NAME))
    assert gen._CACHE_NAME == "generation_cache_v2.csv"
    assert gen._load_cache().empty


def test_bundled_seed_is_used_until_a_disk_cache_exists(tmp_path, monkeypatch):
    seed = tmp_path / "bundled.csv"
    seed.write_text("date,hard_coal_gwh,lignite_gwh,extra\n2026-03-02,96.0,240.0,x\n", encoding="utf-8")
    disk = tmp_path / "persist" / gen._CACHE_NAME
    monkeypatch.setattr(gen, "_CACHE_CSV", str(disk))
    monkeypatch.setattr(gen, "_SEED_CSV", str(seed))

    loaded = gen._load_cache()
    assert list(loaded.columns) == gen._CACHE_COLUMNS
    assert loaded["lignite_gwh"].tolist() == [240.0]

    disk.parent.mkdir()
    disk.write_text("date,hard_coal_gwh,lignite_gwh\n2026-03-03,90.0,200.0\n", encoding="utf-8")
    assert gen._load_cache()["lignite_gwh"].tolist() == [200.0]


def test_unreadable_disk_cache_falls_back_to_the_seed(tmp_path, monkeypatch):
    disk = tmp_path / gen._CACHE_NAME
    disk.write_text("not,a,cache\n", encoding="utf-8")
    seed = tmp_path / "seed.csv"
    seed.write_text("date,hard_coal_gwh,lignite_gwh\n2026-03-02,96.0,240.0\n", encoding="utf-8")
    monkeypatch.setattr(gen, "_CACHE_CSV", str(disk))
    monkeypatch.setattr(gen, "_SEED_CSV", str(seed))
    assert gen._load_cache()["hard_coal_gwh"].tolist() == [96.0]


# ── Incremental refresh ───────────────────────────────────────────────────────

@pytest.fixture
def seeded(tmp_path, monkeypatch):
    """A seed that ends three days ago with stale figures for its last week."""
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    dates = pd.date_range(today - pd.Timedelta(days=400), today - pd.Timedelta(days=3), freq="D")
    seed = pd.DataFrame({"date": dates, "hard_coal_gwh": 1.0, "lignite_gwh": 2.0})
    seed_path = tmp_path / "seed.csv"
    seed.to_csv(seed_path, index=False)
    disk = tmp_path / "persist" / gen._CACHE_NAME
    disk.parent.mkdir()
    client = _FakeClient()
    monkeypatch.setattr(gen, "_SEED_CSV", str(seed_path))
    monkeypatch.setattr(gen, "_CACHE_CSV", str(disk))
    monkeypatch.setattr(gen, "_client", lambda: client)
    gen.fetch_coal_generation.clear()
    yield {"today": today, "disk": disk, "client": client, "last_seed": dates[-1]}
    gen.fetch_coal_generation.clear()


def test_refresh_starts_from_the_seed_and_fetches_only_the_last_days(seeded):
    result = gen.fetch_coal_generation()

    # One request, from a week before the last cached day up to the start of today.
    assert len(seeded["client"].calls) == 1
    start, end = seeded["client"].calls[0]
    assert start == pd.Timestamp(seeded["last_seed"], tz="UTC") - pd.Timedelta(days=7)
    assert end == pd.Timestamp(seeded["today"], tz="UTC")

    saved = pd.read_csv(seeded["disk"], parse_dates=["date"])
    assert saved["date"].is_monotonic_increasing and saved["date"].is_unique
    assert saved["date"].iloc[-1] == seeded["today"] - pd.Timedelta(days=1)
    # Overlapping days take the fresh figures; older days keep the cached ones.
    fresh = saved[saved["date"] >= seeded["last_seed"] - pd.Timedelta(days=7)]
    assert set(fresh["lignite_gwh"]) == {240.0}
    assert set(saved.loc[saved["date"] < fresh["date"].min(), "lignite_gwh"]) == {2.0}

    assert result["source"] == "ENTSO-E"
    assert result["daily_90d"]["lignite_gwh"].iloc[-1] == 240.0


def test_without_a_key_the_seed_is_still_shown(seeded, monkeypatch):
    monkeypatch.setattr(gen, "_client", lambda: None)
    result = gen.fetch_coal_generation()
    assert not result["quarterly"].empty
    assert result["daily_90d"]["lignite_gwh"].iloc[-1] == 2.0
    assert not seeded["disk"].exists()
