"""
Gas storage visualisation: current year versus historical min/mean/max band.
"""
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from data.gas_storage import (
    build_seasonal_bands, calendar_doy, storage_season, EU_TARGET_PCT, EU_TARGET_LABEL,
)
from config.settings import GIE_FAILURE_TTL_S
from utils.helpers import has_agsi_key
from utils.theme import CHART, PLOTLY_TEMPLATE


COLORS = {
    "band_fill": "rgba(100,149,237,0.12)",
    "band_mean": "rgba(100,149,237,0.55)",
    "band_min":  "rgba(220,140,30,0.70)",
    "current":   "#c0392b",
    "mandate":   "rgba(210,180,40,0.55)",
    "grid":      CHART["grid"],
    "text":      CHART["text"],
    "bg":        CHART["bg"],
    "panel_bg":  CHART["panel_bg"],
}


def render_storage_chart(storage_data: dict, region: str = "europe"):
    import streamlit as st

    df = storage_data.get(region, pd.DataFrame())
    fetched_at = storage_data.get("fetched_at")

    label = "Europe" if region == "europe" else "Germany"
    st.subheader(f"Gas Storage: {label}")

    if df.empty:
        if has_agsi_key():
            st.warning(
                f"Gas storage data unavailable for {label}: GIE AGSI+ did not answer. "
                f"The panel retries within {max(GIE_FAILURE_TTL_S // 60, 1)} minutes."
            )
        else:
            st.warning(
                f"Gas storage data unavailable for {label}. "
                "Add AGSI_API_KEY to .env (free registration at agsi.gie.eu)."
            )
        st.caption("Source: Gas Infrastructure Europe (GIE), AGSI+ | agsi.gie.eu")
        return

    current_year = datetime.utcnow().year
    df_current = df[df["gasDayStart"].dt.year == current_year].copy()
    df_current["doy"] = calendar_doy(df_current["gasDayStart"])

    bands = build_seasonal_bands(df, value_col="full")

    if bands.empty:
        st.warning(
            "Historical seasonal bands unavailable: the AGSI+ history holds no earlier "
            "year to compare with."
        )
        return

    # Map the leap-year day of year (see calendar_doy) to dates in a leap reference year
    ref_year = 2024
    bands["date_ref"] = pd.to_datetime(
        bands["day_of_year"].apply(lambda d: f"{ref_year}-{d:03d}"),
        format="%Y-%j",
        errors="coerce",
    )
    df_current["date_ref"] = pd.to_datetime(
        df_current["doy"].apply(lambda d: f"{ref_year}-{d:03d}"),
        format="%Y-%j",
        errors="coerce",
    )

    fig = go.Figure()

    # Min-max shaded band
    fig.add_trace(go.Scatter(
        x=pd.concat([bands["date_ref"], bands["date_ref"][::-1]]),
        y=pd.concat([bands["max"], bands["min"][::-1]]),
        fill="toself",
        fillcolor=COLORS["band_fill"],
        line=dict(color="rgba(0,0,0,0)"),
        name="5-year range",
        hoverinfo="skip",
    ))

    # 5-year average
    fig.add_trace(go.Scatter(
        x=bands["date_ref"],
        y=bands["mean"],
        line=dict(color=COLORS["band_mean"], width=1.5, dash="dot"),
        name="5-year average",
        hovertemplate="5yr avg: %{y:.1f}%<extra></extra>",
    ))

    # 5-year minimum
    fig.add_trace(go.Scatter(
        x=bands["date_ref"],
        y=bands["min"],
        line=dict(color=COLORS["band_min"], width=1),
        name="5-year minimum",
        hovertemplate="5yr min: %{y:.1f}%<extra></extra>",
    ))

    # Current year
    fig.add_trace(go.Scatter(
        x=df_current["date_ref"],
        y=df_current["full"],
        line=dict(color=COLORS["current"], width=2.5),
        name=str(current_year),
        hovertemplate=f"{current_year}: %{{y:.1f}}%<extra></extra>",
    ))

    # EU filling target: 90% at any point between 1 October and 1 December
    fig.add_vrect(
        x0=f"{ref_year}-10-01", x1=f"{ref_year}-12-01",
        fillcolor="rgba(210,180,40,0.06)", line_width=0, layer="below",
    )
    fig.add_hline(
        y=EU_TARGET_PCT,
        line_dash="dash",
        line_color=COLORS["mandate"],
        line_width=1.2,
        annotation_text=EU_TARGET_LABEL,
        annotation_position="top right",
        annotation_font=dict(size=10, color=COLORS["mandate"]),
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            tickformat="%b",
            showgrid=True,
            gridcolor=COLORS["grid"],
            title=None,
        ),
        yaxis=dict(
            title="Fill level (%)",
            ticksuffix="%",
            showgrid=True,
            gridcolor=COLORS["grid"],
            range=[0, 105],
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

    # Inline insight
    latest = df_current.dropna(subset=["full"]).iloc[-1] if not df_current.empty else None
    if latest is not None:
        current_pct = latest["full"]
        min_at_doy = bands.loc[bands["day_of_year"] == latest["doy"], "min"]
        below_min = (not min_at_doy.empty) and (current_pct < min_at_doy.values[0])
        if below_min:
            st.error(
                f"Storage at {current_pct:.1f}%, below the 5-year minimum for this date "
                f"({storage_season(latest['gasDayStart'])})."
            )
        else:
            mean_at_doy = bands.loc[bands["day_of_year"] == latest["doy"], "mean"]
            diff = current_pct - (mean_at_doy.values[0] if not mean_at_doy.empty else current_pct)
            direction = "above" if diff >= 0 else "below"
            st.caption(
                f"Storage at {current_pct:.1f}%, {abs(diff):.1f}pp {direction} the 5-year average."
            )

    if fetched_at:
        st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption("Source: Gas Infrastructure Europe (GIE), AGSI+ | agsi.gie.eu")
