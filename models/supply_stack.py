"""
German merit order / supply stack model.

Represents the structural German generation portfolio ordered by short-run
marginal cost (SRMC). Gas CCGT cost is dynamic (live TTF). Carbon cost is
dynamic (EUA via yfinance). All other costs are static from market averages.
Capacities from Bundesnetzagentur Kraftwerksliste 2024 (rounded, in-service).

Output: merit order step chart, marginal fuel annotation, theoretical implied
price vs NL day-ahead actual, scarcity premium KPI card.

Assumptions logged in methodology expander on the page:
- Merit-order dispatch only; no ramp constraints, reserve requirements, or
  cross-border import corrections.
- Wind and solar dispatched at zero marginal cost and shown at the left of
  the stack (must-run, ahead of all thermal plant).
- All capacities are installed estimates, not available capacity; plant
  availability typically ~85–90% for thermal, ~100% for renewables.
- Hard coal fuel cost estimated at ~11 EUR/MWh_thermal (API2 reference ~80
  USD/t, calorific value 6.98 MWh/t, EUR/USD 0.93).
- EUA emission intensities: gas 0.37 tCO2/MWh_el (CCGT 55% eff),
  coal 0.85 tCO2/MWh_el (hard coal 40% eff).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

try:
    import yfinance as yf
    _YFINANCE = True
except ImportError:
    _YFINANCE = False


# ── Merit order definition ────────────────────────────────────────────────────
# Source: Bundesnetzagentur Kraftwerksliste 2024, rounded installed GW.
# Germany exited nuclear in April 2023; lignite retirement ongoing.
# Gas fleet includes CCGT and open-cycle gas turbine (OCGT) — blended average.

_FUELS: list[tuple] = [
    # (key, display_label, capacity_gw, color)
    ("wind",    "Wind (onshore + offshore)", 66.0, "#3fb950"),
    ("solar",   "Solar PV",                  73.0, "#f0e040"),
    ("hydro",   "Hydro run-of-river",          4.5, "#58a6ff"),
    ("biomass", "Biomass / biogas",             8.5, "#a371f7"),
    ("lignite", "Lignite",                    17.0, "#8b6914"),
    ("coal",    "Hard coal",                  23.0, "#6e7681"),
    ("gas",     "Gas CCGT / OCGT",            30.0, "#e07b39"),
    ("oil",     "Oil / gas turbine (peaker)",  2.5, "#f85149"),
]

# Static variable costs where fuel is not dynamically priced (EUR/MWh_el)
_STATIC_SRMC: dict[str, float] = {
    "wind":    0.0,
    "solar":   0.0,
    "hydro":   0.0,
    "biomass": 28.0,
    "lignite": 20.0,
    "oil":     150.0,
}

# Gas and coal parameters
_GAS_EFFICIENCY  = 0.55      # CCGT thermal efficiency (LHV)
_COAL_EFFICIENCY = 0.40      # Hard coal steam plant efficiency (LHV)
_GAS_EMISSIONS   = 0.37      # tCO2 per MWh_el (CCGT)
_COAL_EMISSIONS  = 0.85      # tCO2 per MWh_el (hard coal)
_COAL_FUEL_COST  = 11.0      # EUR/MWh_thermal (API2 reference, approximate)

_EUA_FALLBACK    = 65.0      # EUR/tonne CO2 — used when yfinance unavailable


# ── EUA price fetcher ─────────────────────────────────────────────────────────

def fetch_eua_price() -> tuple[float, str]:
    """
    Fetch EUA carbon allowance spot price in EUR/tonne CO2 via yfinance.
    Tries CO2.L (ICE EUA futures) then EUAc1=F. Falls back to hardcoded
    estimate if both fail or yfinance is unavailable.

    Returns
    -------
    (price_eur_t, source_label)
    """
    if _YFINANCE:
        for ticker in ("CO2.L", "EUAc1=F"):
            try:
                info = yf.Ticker(ticker).fast_info
                px = float(info.last_price)
                if 5.0 < px < 250.0:
                    return px, ticker
            except Exception:
                continue
    return _EUA_FALLBACK, f"static estimate ({_EUA_FALLBACK:.0f} EUR/t)"


# ── Merit order builder ───────────────────────────────────────────────────────

def build_stack(ttf_eur_mwh: float, eua_eur_t: float) -> pd.DataFrame:
    """
    Build the ordered merit order DataFrame.

    Parameters
    ----------
    ttf_eur_mwh : TTF front-month gas price (EUR/MWh_gas).
    eua_eur_t   : EUA carbon allowance price (EUR/tonne CO2).

    Returns
    -------
    pd.DataFrame sorted by marginal cost, with columns:
        fuel, label, capacity_gw, marginal_cost, cum_start, cum_end, color
    """
    rows = []
    for key, label, cap_gw, color in _FUELS:
        if key == "gas":
            cost = ttf_eur_mwh / _GAS_EFFICIENCY + eua_eur_t * _GAS_EMISSIONS
        elif key == "coal":
            cost = _COAL_FUEL_COST / _COAL_EFFICIENCY + eua_eur_t * _COAL_EMISSIONS
        else:
            cost = _STATIC_SRMC[key]
        rows.append({"fuel": key, "label": label, "capacity_gw": cap_gw,
                     "marginal_cost": round(cost, 1), "color": color})

    df = pd.DataFrame(rows).sort_values("marginal_cost", kind="stable").reset_index(drop=True)
    df["cum_start"] = df["capacity_gw"].cumsum().shift(1).fillna(0.0)
    df["cum_end"]   = df["capacity_gw"].cumsum()
    return df


def identify_marginal_fuel(stack_df: pd.DataFrame, demand_gw: float) -> dict:
    """
    Identify the marginal fuel and implied clearing price at a given demand level.

    Returns
    -------
    dict with keys: fuel, label, marginal_cost, utilisation_pct
    """
    for _, row in stack_df.iterrows():
        if row["cum_end"] >= demand_gw:
            util = ((demand_gw - row["cum_start"]) / row["capacity_gw"] * 100.0
                    if row["capacity_gw"] > 0 else 0.0)
            return {
                "fuel":           row["fuel"],
                "label":          row["label"],
                "marginal_cost":  row["marginal_cost"],
                "utilisation_pct": round(util, 1),
            }
    # Demand exceeds total installed capacity — scarcity pricing
    last = stack_df.iloc[-1]
    return {
        "fuel":           last["fuel"],
        "label":          "Demand exceeds installed capacity",
        "marginal_cost":  last["marginal_cost"],
        "utilisation_pct": 100.0,
    }


# ── Chart builder ─────────────────────────────────────────────────────────────

def make_merit_order_figure(
    stack_df: pd.DataFrame,
    demand_gw: float,
    marginal: dict,
    actual_power_price: float | None = None,
) -> go.Figure:
    """
    Plotly merit order chart: capacity (GW) on x-axis, SRMC (EUR/MWh) on y-axis.
    Each fuel type is a coloured bar whose width = installed capacity.

    Parameters
    ----------
    stack_df           : Output of build_stack().
    demand_gw          : Current demand estimate in GW.
    marginal           : Output of identify_marginal_fuel().
    actual_power_price : NL day-ahead price (EUR/MWh) for comparison, or None.
    """
    fig = go.Figure()

    for _, row in stack_df.iterrows():
        center_x    = row["cum_start"] + row["capacity_gw"] / 2.0
        actual_cost = row["marginal_cost"]
        # Zero-cost fuels (wind, solar, hydro) render as invisible y=0 bars.
        # Display at 2 EUR/MWh minimum so they appear as a thin coloured band;
        # the hover and all analytics still use the real SRMC of €0/MWh.
        display_cost = max(actual_cost, 2.0)
        srmc_label   = "€0/MWh (must-run)" if actual_cost == 0.0 else f"€{actual_cost:.1f}/MWh"
        fig.add_trace(go.Bar(
            x=[center_x],
            y=[display_cost],
            width=[row["capacity_gw"]],
            name=row["label"],
            marker=dict(color=row["color"], opacity=0.85,
                        line=dict(color="#0d1117", width=1)),
            hovertemplate=(
                f"<b>{row['label']}</b><br>"
                f"Capacity: {row['capacity_gw']:.0f} GW<br>"
                f"SRMC: {srmc_label}<br>"
                f"Stack position: {row['cum_start']:.0f}–{row['cum_end']:.0f} GW"
                "<extra></extra>"
            ),
        ))

    # Demand line
    fig.add_vline(
        x=demand_gw, line=dict(color="#e6edf3", width=2, dash="dash"),
        annotation_text=f"  Demand: {demand_gw:.0f} GW",
        annotation_font=dict(color="#e6edf3", size=11),
        annotation_position="top right",
    )

    # Implied clearing price (marginal cost horizontal)
    fig.add_hline(
        y=marginal["marginal_cost"],
        line=dict(color="#f0e040", width=1.5, dash="dot"),
        annotation_text=f"Implied: €{marginal['marginal_cost']:.1f}/MWh  ",
        annotation_font=dict(color="#f0e040", size=11),
        annotation_position="top left",
    )

    # NL actual price for comparison
    if actual_power_price is not None and actual_power_price > 0:
        fig.add_hline(
            y=actual_power_price,
            line=dict(color="#58a6ff", width=1.5, dash="dot"),
            annotation_text=f"NL actual: €{actual_power_price:.1f}/MWh  ",
            annotation_font=dict(color="#58a6ff", size=11),
            annotation_position="bottom left",
        )

    # Footnote: explain the 2 EUR/MWh display minimum for zero-cost fuels
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.01,
        text="* Wind / solar / hydro SRMC = €0/MWh — shown at €2/MWh min height for visibility",
        showarrow=False,
        font=dict(size=9, color="#8b949e"),
        xanchor="left", yanchor="bottom",
    )

    total_cap = stack_df["capacity_gw"].sum()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        barmode="overlay",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=10, color="#8b949e"),
        ),
        xaxis=dict(
            title="Installed capacity — cumulative (GW)",
            range=[0, total_cap * 1.02],
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=11, color="#8b949e"),
        ),
        yaxis=dict(
            title="Short-run marginal cost (EUR/MWh)",
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=11, color="#8b949e"),
            rangemode="tozero",
        ),
        margin=dict(l=60, r=20, t=20, b=50),
        height=420,
    )
    return fig
