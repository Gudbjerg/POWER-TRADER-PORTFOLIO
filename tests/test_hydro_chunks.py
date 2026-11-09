"""Hydro history is fetched in yearly chunks and joined without duplicate weeks."""
import pandas as pd

import data.hydro as hydro


class _FakeClient:
    def __init__(self, fail_year=None):
        self.calls = []
        self.fail_year = fail_year

    def query_aggregate_water_reservoirs_and_hydro_storage(self, country_code, start, end):
        self.calls.append((start, end))
        if self.fail_year is not None and start.year == self.fail_year:
            raise TimeoutError("slow")
        weeks = pd.date_range(start, end, freq="7D")
        return pd.Series(50_000_000.0, index=weeks)


def test_yearly_chunks_join(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(hydro, "_client", lambda: fake)
    df = hydro.fetch_hydro_reservoirs(years=3)
    assert len(fake.calls) == 3
    assert df["week_start"].is_unique and df["week_start"].is_monotonic_increasing
    assert df["filling_twh"].eq(50.0).all()
    assert (df["week_start"].max() - df["week_start"].min()).days > 2 * 365
