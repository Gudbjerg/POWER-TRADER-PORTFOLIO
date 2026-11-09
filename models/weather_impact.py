"""
Weather to power: from an ECMWF forecast to German residual load and a day-ahead price.

Chain
-----
1. Weather indices (``data/weather.py``): wind capacity factor, irradiance, heating degree
   days, each a weighted mean over sites where the German fleet and load sit.
2. Fleet calibration: ordinary least squares of what the fleet produced (energy-charts.info)
   on the indices from archived ECMWF forecasts for the same days, so the forecast is read
   in GW of the fleet as it is today rather than through assumed installed capacities.
     wind_gw  = a + b * wind_idx
     solar_gw = a + b * solar_idx
     load_gw  = a + b * hdd + c * saturday + d * sunday
3. Residual load = load - wind - solar: the demand the thermal fleet and imports must meet.
4. Price: DE-LU day-ahead daily average on residual load and the previous TTF close,
     price = a + b * residual_gw + c * ttf
   fitted on actual residual load (the structural relation), then applied to the forecast.

Validation: every equation is fitted on the first part of the sample and the whole chain is
run on the held-out last ``holdout_days`` days from the archived forecasts, against the
persistence benchmark (yesterday's price), the bar a weather model has to clear.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Fit:
    """One least-squares equation: coefficients by regressor name, R2 and sample size."""
    coef: dict[str, float]
    r2: float
    n: int
    mae: float

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        out = pd.Series(self.coef.get("const", 0.0), index=frame.index, dtype=float)
        for name, beta in self.coef.items():
            if name != "const":
                out = out + beta * frame[name]
        return out


def ols(frame: pd.DataFrame, target: str, regressors: list[str]) -> Fit:
    """OLS with intercept on the rows where target and all regressors are present."""
    data = frame[[target] + regressors].dropna()
    if len(data) < len(regressors) + 5:
        raise ValueError(f"too few observations for {target}: {len(data)}")
    X = np.column_stack([np.ones(len(data))] + [data[r].to_numpy(float) for r in regressors])
    y = data[target].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return Fit(coef=dict(zip(["const"] + regressors, map(float, beta))), r2=r2, n=len(data),
               mae=float(np.abs(resid).mean()))


def _calendar(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dow = pd.DatetimeIndex(out.index).dayofweek
    out["saturday"] = (dow == 5).astype(float)
    out["sunday"] = (dow == 6).astype(float)
    return out


def _with_ttf(frame: pd.DataFrame, ttf: pd.DataFrame) -> pd.DataFrame:
    """Attach the TTF close of the previous trading day (known when the auction clears)."""
    out = frame.copy()
    if ttf is None or ttf.empty:
        out["ttf"] = np.nan
        return out
    t = ttf[["date", "price"]].dropna().copy()
    t["date"] = pd.to_datetime(t["date"]).dt.normalize()
    t = t.drop_duplicates("date", keep="last").set_index("date")["price"].sort_index()
    days = pd.DatetimeIndex(out.index)
    out["ttf"] = t.reindex(t.index.union(days)).ffill().shift(1).reindex(days).to_numpy()
    return out


@dataclass
class WeatherModel:
    wind: Fit
    solar: Fit
    load: Fit
    price: Fit
    sample: pd.DataFrame
    validation: dict = field(default_factory=dict)

    def generation(self, indices: pd.DataFrame) -> pd.DataFrame:
        """Fleet wind, solar, load and residual load (GW) from daily weather indices."""
        f = _calendar(indices)
        out = pd.DataFrame(index=f.index)
        out["wind_gw"] = self.wind.predict(f).clip(lower=0.0)
        out["solar_gw"] = self.solar.predict(f).clip(lower=0.0)
        out["load_gw"] = self.load.predict(f)
        out["residual_gw"] = out["load_gw"] - out["wind_gw"] - out["solar_gw"]
        return out

    def price_from(self, residual_gw: pd.Series, ttf: float | pd.Series | None) -> pd.Series:
        frame = pd.DataFrame({"residual_gw": residual_gw})
        frame["ttf"] = ttf if ttf is not None else np.nan
        return self.price.predict(frame)

    @property
    def uses_gas(self) -> bool:
        return "ttf" in self.price.coef

    @property
    def eur_per_gw(self) -> float:
        """Price change (EUR/MWh) for one GW more residual load."""
        return self.price.coef["residual_gw"]

    @property
    def gw_per_wind_idx(self) -> float:
        return self.wind.coef["wind_idx"]


def fit_weather_model(hindcast: pd.DataFrame, actuals: pd.DataFrame, ttf: pd.DataFrame,
                      holdout_days: int = 30) -> WeatherModel:
    """Calibrate the chain on archived forecasts and actuals; validate on the last days.

    ``hindcast``: daily indices from archived forecasts (wind_idx, solar_idx, hdd);
    ``actuals``: daily wind_gw, solar_gw, load_gw, price; ``ttf``: date, price.
    """
    sample = _calendar(hindcast.join(actuals, how="inner"))
    sample = _with_ttf(sample, ttf)
    sample["residual_gw"] = sample["load_gw"] - sample["wind_gw"] - sample["solar_gw"]
    sample = sample.dropna(subset=["wind_idx", "solar_idx", "hdd", "wind_gw", "solar_gw", "load_gw"])

    # Gas enters the price equation when the TTF history covers the sample; without it
    # the equation runs on residual load alone (the gas level is then in the intercept).
    price_x = ["residual_gw", "ttf"] if sample["ttf"].notna().sum() >= 40 else ["residual_gw"]

    def fit_all(frame: pd.DataFrame) -> tuple[Fit, Fit, Fit, Fit]:
        return (ols(frame, "wind_gw", ["wind_idx"]),
                ols(frame, "solar_gw", ["solar_idx"]),
                ols(frame, "load_gw", ["hdd", "saturday", "sunday"]),
                ols(frame, "price", price_x))

    validation: dict = {}
    if len(sample) >= holdout_days + 40:
        train, test = sample.iloc[:-holdout_days], sample.iloc[-holdout_days:]
        w, s, l, p = fit_all(train)
        trial = WeatherModel(w, s, l, p, train)
        gen = trial.generation(test)
        pred = trial.price_from(gen["residual_gw"], test["ttf"].ffill())
        actual = test["price"]
        persistence = sample["price"].shift(1).reindex(test.index)
        ok = actual.notna() & pred.notna() & persistence.notna()
        if ok.sum() >= 10:
            err_model = (pred[ok] - actual[ok]).abs()
            err_pers = (persistence[ok] - actual[ok]).abs()
            # Direction of the day-on-day move, the call a trader acts on
            prev = sample["price"].shift(1).reindex(test.index)[ok]
            hit = float((np.sign(pred[ok] - prev) == np.sign(actual[ok] - prev)).mean())
            validation = {
                "start": test.index[ok][0], "end": test.index[ok][-1], "n": int(ok.sum()),
                "mae_model": float(err_model.mean()), "mae_persistence": float(err_pers.mean()),
                "direction_hit": hit,
                "wind_mae_gw": float((gen["wind_gw"] - test["wind_gw"]).abs().mean()),
                "frame": pd.DataFrame({"actual": actual, "model": pred, "persistence": persistence}),
            }
    w, s, l, p = fit_all(sample)
    return WeatherModel(w, s, l, p, sample, validation)


def ensemble_band(model: WeatherModel, members: pd.DataFrame, base: pd.DataFrame,
                  ttf: float) -> pd.DataFrame:
    """P10/P50/P90 of wind GW and price across ECMWF ensemble members.

    ``members``: daily wind index per member; ``base``: the deterministic generation frame
    (load and solar are held at the deterministic forecast, so the band is wind-only).
    """
    days = members.index.intersection(base.index)
    if len(days) == 0:
        return pd.DataFrame()
    wind = (model.wind.coef["const"] + model.gw_per_wind_idx * members.loc[days]).clip(lower=0.0)
    resid = wind.rsub(base.loc[days, "load_gw"] - base.loc[days, "solar_gw"], axis=0)
    price = model.price.coef["const"] + model.eur_per_gw * resid + model.price.coef.get("ttf", 0.0) * ttf
    q = lambda f, p: f.quantile(p, axis=1)
    return pd.DataFrame({
        "wind_p10": q(wind, 0.1), "wind_p50": q(wind, 0.5), "wind_p90": q(wind, 0.9),
        "price_p10": q(price, 0.1), "price_p50": q(price, 0.5), "price_p90": q(price, 0.9),
    })


def revision_impact(model: WeatherModel, revision: pd.DataFrame) -> pd.DataFrame:
    """Overnight change in the wind outlook (GW) and its price effect (EUR/MWh) per day.

    More wind lowers residual load one for one, so the price effect is
    -(EUR per GW of residual load) x (change in wind GW).
    """
    if revision.empty:
        return pd.DataFrame()
    slope = model.gw_per_wind_idx
    out = pd.DataFrame(index=revision.index)
    out["wind_latest_gw"] = (model.wind.coef["const"] + slope * revision["latest"]).clip(lower=0.0)
    out["wind_previous_gw"] = (model.wind.coef["const"] + slope * revision["previous"]).clip(lower=0.0)
    out["wind_change_gw"] = out["wind_latest_gw"] - out["wind_previous_gw"]
    out["price_effect"] = -model.eur_per_gw * out["wind_change_gw"]
    return out


def hydro_anomaly(outlook: pd.DataFrame, normals: pd.DataFrame, climate_years: str = "2016 to 2025") -> dict:
    """Catchment precipitation and temperature over the outlook against calendar-day normals.

    ``normals`` is indexed by month and day ("MM-DD", 29 February folded onto 28 February).
    Precipitation on days below 0.5 C is counted as snow: it adds to the snowpack and reaches
    the reservoirs only at the spring melt.
    """
    if outlook.empty or normals.empty:
        return {}
    days = outlook.index.strftime("%m-%d").str.replace("02-29", "02-28")
    norm = normals.reindex(days)
    norm.index = outlook.index
    precip = float(outlook["precip_mm"].sum())
    precip_norm = float(norm["precip_mm"].sum())
    snow = float(outlook.loc[outlook["temp_c"] < 0.5, "precip_mm"].sum())
    return {
        "days": len(outlook),
        "precip_mm": precip, "precip_normal_mm": precip_norm,
        "precip_ratio": precip / precip_norm if precip_norm > 0 else float("nan"),
        "temp_c": float(outlook["temp_c"].mean()), "temp_normal_c": float(norm["temp_c"].mean()),
        "snow_share": snow / precip if precip > 0 else 0.0,
        "daily": pd.DataFrame({"precip_mm": outlook["precip_mm"], "normal_mm": norm["precip_mm"],
                               "temp_c": outlook["temp_c"], "temp_normal_c": norm["temp_c"]}),
        "climate_years": climate_years,
    }


def load_weather_model() -> WeatherModel:
    """The calibrated chain on the standard 120-day window, cached for six hours.

    Shared by the Weather Desk and the background warm-up, so both use one cache entry.
    """
    return _load_weather_model_cached()


def _cached(fn):
    try:
        import streamlit as st
        return st.cache_data(ttl=6 * 3600, show_spinner=False)(fn)
    except Exception:
        return fn


@_cached
def _load_weather_model_cached() -> WeatherModel:
    from data import weather as W
    from models.ttf_backtest import fetch_ttf_history
    start, end = W.calibration_window(120)
    return fit_weather_model(W.fetch_hindcast_indices(start, end), W.fetch_de_actuals(start, end),
                             fetch_ttf_history(years=7))
