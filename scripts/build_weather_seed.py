"""
Build the files the Weather Desk ships with, so a fresh server does not spend its
Open-Meteo budget on history that never changes:

* data/weather_hindcast_v1.csv: daily weather indices from archived ECMWF forecasts
  (the calibration sample; the app appends newer days itself);
* data/weather_hydro_normals_v1.csv: ERA5 2016 to 2025 calendar-day normals for the
  Norwegian catchments.

Usage: python -m scripts.build_weather_seed [days]
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta

logging.disable(logging.WARNING)

from data import weather as W  # noqa: E402


def main(days: int = 180) -> None:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    hind = W._weather_request(W.HISTORICAL_URL, {"start_date": start.isoformat(), "end_date": end.isoformat()})
    hind.round(5).to_csv(os.path.join(W._BUNDLED_DIR, W._HINDCAST_NAME), index_label="date")
    print(f"hindcast: {len(hind)} days, {hind.index.min():%Y-%m-%d} to {hind.index.max():%Y-%m-%d}")
    normals = W.normals_from_daily(W.era5_hydro_daily())
    normals.round(3).to_csv(W._NORMALS_SEED)
    print(f"normals: {len(normals)} calendar days")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 180)
