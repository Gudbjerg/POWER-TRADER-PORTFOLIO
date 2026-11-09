"""
Weather Desk: from the ECMWF forecast to German residual load and the DE-LU day-ahead price,
the overnight forecast revisions that move the next auction, and the precipitation outlook
for the Norwegian hydro catchments.

Data: Open-Meteo (ECMWF IFS 0.25 degrees deterministic, 51-member ensemble, single runs,
archived forecasts, ERA5) and energy-charts.info (German fleet output, load, DE-LU prices),
both keyless. Model: models/weather_impact.py.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

load_dotenv()

st.set_page_config(page_title="Weather Desk", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

from data import weather as W
from models import weather_impact as M
from models.ttf_backtest import fetch_ttf_history
from utils.helpers import apply_dark_theme, delta_span, eur, kpi_card
from utils.theme import (ACCENTS, CHART, COLORS, PLOTLY_TEMPLATE, callout, page_footer, page_header,
                         render, section_header, transmission_chain)

apply_dark_theme()

ACCENT = ACCENTS["weather"]
C = {"wind": "#2f7a47", "solar": "#b08d00", "load": COLORS["text"], "resid": ACCENTS["l2"],
     "price": ACCENTS["l5"], "band": "rgba(194,57,31,0.14)", "pers": COLORS["faint"],
     "rain": ACCENT, "normal": COLORS["faint"]}

page_header(
    "Weather Desk", "From the forecast to the price",
    "Wind, sun and cold decide how much demand is left for gas and coal plant in Germany, and that "
    "residual load sets the day-ahead price. This page reads the latest ECMWF forecast through a model "
    "calibrated on the German fleet, shows how far the outlook moved between two runs, and sets the "
    "rain and snow over the Norwegian reservoirs against the normal.",
    sources=["ECMWF IFS via Open-Meteo", "energy-charts.info (SMARD, ENTSO-E)", "Yahoo Finance (TTF)"],
    accent=ACCENT,
)


# ── Data and model ────────────────────────────────────────────────────────────

def _calibrated_model():
    return M.load_weather_model()


@st.cache_data(ttl=3600, show_spinner=False)
def _cleared_prices() -> pd.Series:
    """DE-LU day-ahead daily averages already cleared for today and tomorrow."""
    today = pd.Timestamp.now(tz=W.TZ).tz_localize(None).normalize()
    try:
        frame = W.fetch_de_actuals.__wrapped__(today.date().isoformat(),
                                               (today + pd.Timedelta(days=1)).date().isoformat())
        return frame["price"].dropna()
    except Exception:
        return pd.Series(dtype=float)


def _ttf_now() -> tuple[float | None, str]:
    ttf = fetch_ttf_history(years=7)
    if ttf.empty:
        return None, "TTF unavailable"
    last = ttf.dropna().iloc[-1]
    return float(last["price"]), f"TTF €{last['price']:.2f}/MWh ({pd.Timestamp(last['date']):%d %b})"


with st.spinner("Calibrating the weather model on the German fleet (first load only)…"):
    try:
        model = _calibrated_model()
    except Exception as exc:
        model = None
        render(callout(
            "The weather model could not be calibrated because a source did not answer "
            f"({exc}). The page fills in when Open-Meteo and energy-charts.info respond again.",
            "warn", title="Weather Desk unavailable"))

if model is None:
    page_footer()
    st.stop()

ttf_now, ttf_label = _ttf_now()

try:
    forecast = W.fetch_forecast_indices(10)
    outlook = model.generation(forecast)
    outlook["price"] = model.price_from(outlook["residual_gw"], ttf_now)
except Exception as exc:
    outlook = pd.DataFrame()
    render(callout(f"The ECMWF forecast is unavailable right now ({exc}).", "warn"))

cleared = _cleared_prices()
today = pd.Timestamp.now(tz=W.TZ).tz_localize(None).normalize()
tomorrow = today + pd.Timedelta(days=1)

# ── Tomorrow as a chain ───────────────────────────────────────────────────────

if not outlook.empty and tomorrow in outlook.index:
    row = outlook.loc[tomorrow]
    fc_row = forecast.loc[tomorrow]
    renew = row["wind_gw"] + row["solar_gw"]
    price_tone = "power"
    cleared_note = (f"cleared at €{cleared[tomorrow]:,.0f}" if tomorrow in cleared.index
                    else "auction clears 12:00 CET")
    render(transmission_chain(
        [
            {"label": "Wind at hub height", "value": f"{fc_row['wind_idx'] * 100:.0f}%",
             "note": f"capacity factor, {fc_row['temp_c']:.0f} °C mean"},
            {"label": "Wind and solar output", "value": f"{renew:,.1f} GW",
             "note": f"wind {row['wind_gw']:.1f}, solar {row['solar_gw']:.1f}"},
            {"label": "Residual load", "value": f"{row['residual_gw']:,.1f} GW",
             "note": f"of {row['load_gw']:.1f} GW demand"},
            {"label": "DE-LU day-ahead, model", "value": f"€{row['price']:,.0f}", "tone": price_tone,
             "note": cleared_note},
        ],
        caption=(f"Tomorrow, {tomorrow:%A %d %B}: daily averages from the latest ECMWF run, read through "
                 f"the fleet calibration below. Gas held at the last {ttf_label}."),
    ))

# ── Meters ────────────────────────────────────────────────────────────────────

try:
    revision = M.revision_impact(model, W.fetch_wind_revision(7))
    rev_runs = W.fetch_wind_revision(7).attrs
except Exception as exc:
    revision, rev_runs = pd.DataFrame(), {}
    rev_error = str(exc)
else:
    rev_error = ""

try:
    hydro = M.hydro_anomaly(W.fetch_hydro_outlook(10), W.fetch_hydro_normals(),
                            f"{W.CLIMATE_YEARS[0]} to {W.CLIMATE_YEARS[1]}")
except Exception as exc:
    hydro = {}
    hydro_error = str(exc)
else:
    hydro_error = ""

if not outlook.empty:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        if tomorrow in outlook.index and today in outlook.index:
            d = outlook.loc[tomorrow, "price"] - outlook.loc[today, "price"]
            st.markdown(kpi_card("Tomorrow vs today, model", f"{eur(d, signed=True)}/MWh",
                                 delta_span(f"residual load {outlook.loc[tomorrow, 'residual_gw'] - outlook.loc[today, 'residual_gw']:+.1f} GW",
                                            "red" if d > 0 else "green"), accent=ACCENT),
                        unsafe_allow_html=True)
    with k2:
        lo = outlook["price"].idxmin()
        st.markdown(kpi_card("Cheapest day in ten", f"{lo:%a %d %b}",
                             delta_span(f"€{outlook.loc[lo, 'price']:,.0f}/MWh, wind {outlook.loc[lo, 'wind_gw']:.0f} GW", "green"),
                             accent=ACCENT), unsafe_allow_html=True)
    with k3:
        if not revision.empty:
            big = revision["price_effect"].abs().idxmax()
            eff = revision.loc[big, "price_effect"]
            st.markdown(kpi_card("Largest overnight revision", f"{eur(eff, signed=True)}/MWh",
                                 delta_span(f"{big:%a %d %b}: wind {revision.loc[big, 'wind_change_gw']:+.1f} GW",
                                            "red" if eff > 0 else "green"), accent=ACCENT),
                        unsafe_allow_html=True)
        else:
            st.markdown(kpi_card("Largest overnight revision", "No run pair",
                                 delta_span("ECMWF run archive did not answer", "amber"), accent=ACCENT),
                        unsafe_allow_html=True)
    with k4:
        if hydro:
            ratio = hydro["precip_ratio"]
            st.markdown(kpi_card("Reservoir catchments, 10 days", f"{ratio * 100:.0f}% of normal",
                                 delta_span(f"{hydro['precip_mm']:.0f} mm vs {hydro['precip_normal_mm']:.0f} mm; "
                                            f"{hydro['snow_share'] * 100:.0f}% as snow",
                                            "green" if ratio >= 1 else "red"), accent=ACCENT),
                        unsafe_allow_html=True)
        else:
            st.markdown(kpi_card("Reservoir catchments, 10 days", "No forecast",
                                 delta_span("Open-Meteo did not answer", "amber"), accent=ACCENT),
                        unsafe_allow_html=True)

# ── Ten-day outlook ───────────────────────────────────────────────────────────

section_header("Ten-day outlook",
               "Fleet output and demand (GW, daily averages) and the price the calibrated model "
               "reads from them, with the 10th to 90th percentile of the 51 ECMWF ensemble members.")

if not outlook.empty:
    try:
        band = M.ensemble_band(model, W.fetch_wind_ensemble(10), outlook, ttf_now or 0.0)
    except Exception:
        band = pd.DataFrame()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.55, 0.45])
    fig.add_trace(go.Bar(x=outlook.index, y=outlook["wind_gw"], name="Wind", marker_color=C["wind"],
                         hovertemplate="Wind %{y:.1f} GW<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=outlook.index, y=outlook["solar_gw"], name="Solar", marker_color=C["solar"],
                         hovertemplate="Solar %{y:.1f} GW<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=outlook.index, y=outlook["load_gw"], name="Demand", mode="lines+markers",
                             line=dict(color=C["load"], width=2),
                             hovertemplate="Demand %{y:.1f} GW<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=outlook.index, y=outlook["residual_gw"], name="Residual load",
                             mode="lines+markers", line=dict(color=C["resid"], width=2, dash="dot"),
                             hovertemplate="Residual %{y:.1f} GW<extra></extra>"), row=1, col=1)
    if not band.empty:
        fig.add_trace(go.Scatter(x=band.index, y=band["price_p90"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(x=band.index, y=band["price_p10"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=C["band"], name="Ensemble P10 to P90",
                                 hovertemplate="P10 €%{y:,.0f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=outlook.index, y=outlook["price"], name="Model price (deterministic run)",
                             mode="lines+markers", line=dict(color=C["price"], width=2.4),
                             hovertemplate="Model €%{y:,.0f}/MWh<extra></extra>"), row=2, col=1)
    if not cleared.empty:
        fig.add_trace(go.Scatter(x=cleared.index, y=cleared.values, name="Cleared day-ahead",
                                 mode="markers", marker=dict(color=COLORS["text"], size=11, symbol="diamond"),
                                 hovertemplate="Cleared €%{y:,.0f}/MWh<extra></extra>"), row=2, col=1)
    fig.update_layout(template=PLOTLY_TEMPLATE, barmode="stack", height=560,
                      paper_bgcolor=CHART["bg"], plot_bgcolor=CHART["panel_bg"],
                      margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="GW", row=1, col=1)
    fig.update_yaxes(title_text="DE-LU €/MWh", row=2, col=1)
    st.plotly_chart(fig, width="stretch")

    table = outlook[["wind_gw", "solar_gw", "load_gw", "residual_gw", "price"]].copy()
    if not band.empty:
        table["band"] = [f"€{a:,.0f} to €{b:,.0f}" for a, b in zip(band["price_p10"].reindex(table.index),
                                                                   band["price_p90"].reindex(table.index))]
    table.index = pd.Index(table.index.strftime("%a %d %b"), name="Day")
    table = table.rename(columns={"wind_gw": "Wind GW", "solar_gw": "Solar GW", "load_gw": "Demand GW",
                                  "residual_gw": "Residual GW", "price": "Model €/MWh",
                                  "band": "Ensemble P10 to P90"})
    st.dataframe(table.style.format({c: "{:,.1f}" for c in table.columns if c != "Ensemble P10 to P90"}),
                 width="stretch")

    swing = outlook["price"].max() - outlook["price"].min()
    windy = outlook["wind_gw"].idxmax()
    render(callout(
        f"Over the ten days the model price spans €{swing:,.0f}/MWh. The windiest day is "
        f"{windy:%A %d %B} ({outlook.loc[windy, 'wind_gw']:.0f} GW), when residual load falls to "
        f"{outlook.loc[windy, 'residual_gw']:.0f} GW. Every GW of residual load moves the modelled price by "
        f"€{model.eur_per_gw:.2f}/MWh. Beyond about five days the ensemble band widens and the deterministic "
        "run is one outcome among many; read the band, not the line.", "info"))

# ── Overnight revisions ───────────────────────────────────────────────────────

section_header("What moved overnight",
               "The newest ECMWF run against the run 24 hours earlier, for the same days. A revision "
               "that lands before the 12:00 CET day-ahead gate closure is information the auction has "
               "not yet cleared on.")

if not revision.empty:
    fig_r = go.Figure()
    fig_r.add_trace(go.Bar(
        x=revision.index, y=revision["price_effect"],
        marker_color=[C["price"] if v > 0 else C["wind"] for v in revision["price_effect"]],
        customdata=np.column_stack([revision["wind_change_gw"], revision["wind_previous_gw"],
                                    revision["wind_latest_gw"]]),
        hovertemplate=("%{x|%a %d %b}: price effect %{y:+,.1f} €/MWh<br>wind %{customdata[1]:.1f} → "
                       "%{customdata[2]:.1f} GW (%{customdata[0]:+.1f})<extra></extra>"),
        name="Price effect of the wind revision"))
    fig_r.add_hline(y=0, line_color=CHART["zero"], line_width=1)
    fig_r.update_layout(template=PLOTLY_TEMPLATE, height=280, paper_bgcolor=CHART["bg"],
                        plot_bgcolor=CHART["panel_bg"], margin=dict(l=10, r=10, t=10, b=10),
                        yaxis_title="€/MWh", showlegend=False)
    st.plotly_chart(fig_r, width="stretch")
    lr, pr = rev_runs.get("latest_run"), rev_runs.get("previous_run")
    runs = (f"Runs compared: {lr:%d %b %H:%M} UTC against {pr:%d %b %H:%M} UTC. " if lr is not None else "")
    up = revision["price_effect"].idxmax()
    dn = revision["price_effect"].idxmin()
    render(callout(
        runs + f"The largest upward price revision is {up:%A %d %B} "
        f"({eur(revision.loc[up, 'price_effect'], signed=True)}/MWh as wind moved {revision.loc[up, 'wind_change_gw']:+.1f} GW), "
        f"the largest downward {dn:%A %d %B} ({eur(revision.loc[dn, 'price_effect'], signed=True)}/MWh as wind moved "
        f"{revision.loc[dn, 'wind_change_gw']:+.1f} GW). "
        "ECMWF's 00 UTC run is normally published between about 06:00 and 08:00 CET, ahead of the gate "
        "closure; a trader reading it then is reacting to public information sooner, not trading ahead of "
        "anyone's orders. The run archive used here lags the live run by some hours.", "info"))
else:
    render(callout(f"The ECMWF run archive did not answer ({rev_error or 'no data'}), so no revision is shown.",
                   "warn"))

# ── Nordic hydro ──────────────────────────────────────────────────────────────

section_header("Rain and snow over the Norwegian reservoirs",
               "Precipitation over five western catchments behind NO2 and NO5, against the ERA5 "
               "2016 to 2025 normal for the same calendar days.")

if hydro:
    daily = hydro["daily"]
    fig_h = go.Figure()
    fig_h.add_trace(go.Bar(x=daily.index, y=daily["precip_mm"], name="Forecast precipitation",
                           marker_color=C["rain"], hovertemplate="%{y:.1f} mm<extra></extra>"))
    fig_h.add_trace(go.Scatter(x=daily.index, y=daily["normal_mm"], name="Normal (ERA5 2016 to 2025)",
                               mode="lines+markers", line=dict(color=C["normal"], width=2, dash="dot"),
                               hovertemplate="Normal %{y:.1f} mm<extra></extra>"))
    fig_h.add_trace(go.Scatter(x=daily.index, y=daily["temp_c"], name="Mean temperature", yaxis="y2",
                               mode="lines", line=dict(color=C["price"], width=1.6),
                               hovertemplate="%{y:.1f} °C<extra></extra>"))
    fig_h.update_layout(template=PLOTLY_TEMPLATE, height=300, paper_bgcolor=CHART["bg"],
                        plot_bgcolor=CHART["panel_bg"], margin=dict(l=10, r=10, t=10, b=10),
                        yaxis=dict(title="mm per day"), hovermode="x unified",
                        yaxis2=dict(title="°C", overlaying="y", side="right", showgrid=False),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_h, width="stretch")
    ratio = hydro["precip_ratio"]
    if ratio >= 1.15:
        lean = ("wetter than normal: inflow adds to the hydro balance, which tends to widen the discount "
                "of southern Norway to the Continent")
    elif ratio <= 0.85:
        lean = ("drier than normal: the hydro balance weakens, producers value the water higher and "
                "southern Norway tends to price closer to the Continent")
    else:
        lean = "close to normal: no strong push on the hydro balance from precipitation"
    snow = hydro["snow_share"]
    render(callout(
        f"The next {hydro['days']} days bring {hydro['precip_mm']:.0f} mm against a normal of "
        f"{hydro['precip_normal_mm']:.0f} mm ({ratio * 100:.0f}%), {lean}. "
        f"{snow * 100:.0f}% falls on days below 0.5 °C and is stored as snow, which reaches the reservoirs "
        f"only at the spring melt. Mean temperature {hydro['temp_c']:.1f} °C against a normal of "
        f"{hydro['temp_normal_c']:.1f} °C. Catchment points are grid cells, lower than the high mountains, "
        "so the snow share is a lower bound.", "info"))
else:
    render(callout(f"The catchment forecast is unavailable ({hydro_error or 'no data'}).", "warn"))

# ── Model credibility ─────────────────────────────────────────────────────────

section_header("How far to trust it",
               "Each equation is fitted on 120 days of archived ECMWF forecasts against what the fleet "
               "did. The full chain is then tested on the last 30 days it was not fitted on.")

v = model.validation
if v:
    c1, c2, c3, c4 = st.columns(4)
    better = v["mae_model"] < v["mae_persistence"]
    with c1:
        st.markdown(kpi_card("Price error, model", f"€{v['mae_model']:.1f}",
                             delta_span("mean absolute error per day", "blue"), accent=ACCENT),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Price error, yesterday's price", f"€{v['mae_persistence']:.1f}",
                             delta_span("persistence benchmark", "blue"), accent=ACCENT),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Direction of the daily move", f"{v['direction_hit'] * 100:.0f}%",
                             delta_span(f"of {v['n']} days called correctly", "green" if v["direction_hit"] > 0.5 else "red"),
                             accent=ACCENT), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Wind error", f"{v['wind_mae_gw']:.1f} GW",
                             delta_span("daily mean, archived forecast vs fleet", "blue"), accent=ACCENT),
                    unsafe_allow_html=True)
    vf = v["frame"].dropna()
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=vf.index, y=vf["actual"], name="Cleared DE-LU", mode="lines+markers",
                               line=dict(color=COLORS["text"], width=2)))
    fig_v.add_trace(go.Scatter(x=vf.index, y=vf["model"], name="Weather model", mode="lines+markers",
                               line=dict(color=C["price"], width=2)))
    fig_v.add_trace(go.Scatter(x=vf.index, y=vf["persistence"], name="Yesterday's price", mode="lines",
                               line=dict(color=C["pers"], width=1.5, dash="dot")))
    fig_v.update_layout(template=PLOTLY_TEMPLATE, height=300, paper_bgcolor=CHART["bg"],
                        plot_bgcolor=CHART["panel_bg"], margin=dict(l=10, r=10, t=10, b=10),
                        yaxis_title="€/MWh", hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_v, width="stretch")
    verdict = (f"Out of sample ({v['start']:%d %b} to {v['end']:%d %b %Y}) the model's daily error is "
               f"€{v['mae_model']:.1f}/MWh against €{v['mae_persistence']:.1f}/MWh for yesterday's price"
               + (", so the weather explains day-to-day moves that persistence misses. " if better else
                  ", so it does not beat persistence over this window. ")
               + "The test uses archived short-lead forecasts, so it measures the model at a one-day "
                 "horizon; errors grow with lead time, which the ensemble band shows.")
    render(callout(verdict, "ok" if better else "warn"))

fits = pd.DataFrame([
    {"Equation": "Wind GW", "Regressors": "wind index", "R²": model.wind.r2, "MAE": model.wind.mae, "Days": model.wind.n},
    {"Equation": "Solar GW", "Regressors": "irradiance", "R²": model.solar.r2, "MAE": model.solar.mae, "Days": model.solar.n},
    {"Equation": "Demand GW", "Regressors": "heating degree days, Saturday, Sunday", "R²": model.load.r2,
     "MAE": model.load.mae, "Days": model.load.n},
    {"Equation": "DE-LU €/MWh", "Regressors": "residual load" + (", TTF" if model.uses_gas else ""),
     "R²": model.price.r2, "MAE": model.price.mae, "Days": model.price.n},
]).set_index("Equation")
st.dataframe(fits.style.format({"R²": "{:.2f}", "MAE": "{:.2f}"}), width="stretch")

with st.expander("Method, assumptions and limits", expanded=False):
    gas = (f"€{model.price.coef['ttf']:.2f}/MWh of power per €1/MWh of TTF" if model.uses_gas
           else "not estimated (TTF unavailable), so the gas level sits in the intercept")
    st.markdown(f"""
**Weather indices.** Hub-height (100 m) wind at ten sites weighted to where German capacity
stands (onshore north and east, North Sea and Baltic offshore), converted to a capacity factor
with a generic power curve (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s). Global irradiance at
six sites for solar. Heating degree days below {W.HDD_BASE_C} °C at five load centres for demand.

**Fleet calibration.** Least squares of energy-charts.info daily output on the indices from
archived ECMWF forecasts, so the slopes absorb today's installed capacity, curtailment and
the forecast's own bias: wind GW = {model.wind.coef['const']:.1f} + {model.gw_per_wind_idx:.1f} × index;
demand falls {abs(model.load.coef['sunday']):.1f} GW on Sundays and rises
{model.load.coef['hdd']:.2f} GW per degree day.

**Price.** DE-LU daily average on actual residual load and the previous TTF close:
€{model.eur_per_gw:.2f}/MWh per GW of residual load; gas pass-through {gas}. Forecasts hold
TTF at the last close.

**Limits.** One national merit order: no interconnector flows, outages, hydro or battery
dispatch. Daily averages hide the evening peak, where scarcity prices form. A linear price
equation understates the convexity at very high residual load. The ensemble band varies wind
only. Revisions come from the run archive, which trails the live run. Figures are estimates
for analysis, not trading advice.
""")

page_footer(
    [
        ("Weather", "ECMWF IFS 0.25° via Open-Meteo (CC BY 4.0)", "https://open-meteo.com"),
        ("Climate normal", "ERA5 via Open-Meteo", "https://open-meteo.com/en/docs/historical-weather-api"),
        ("German fleet and prices", "energy-charts.info, Fraunhofer ISE (CC BY 4.0)", "https://www.energy-charts.info"),
        ("Gas", "Yahoo Finance (TTF=F)"),
    ],
    note="Calibration refreshes every six hours, forecasts every three.",
)
