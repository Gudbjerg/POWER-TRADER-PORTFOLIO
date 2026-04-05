"""
Gas storage visualisation: current year versus historical min/mean/max band.
"""
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from data.gas_storage import build_seasonal_bands


COLORS = {
    "band_fill": "rgba(100,149,237,0.12)",
    "band_mean": "rgba(100,149,237,0.55)",
    "band_min":  "rgba(220,140,30,0.70)",
    "current":   "#c0392b",
    "mandate":   "rgba(210,180,40,0.55)",
    "grid":      "rgba(255,255,255,0.06)",
    "text":      "#c9d1d9",
    "bg":        "#0d1117",
    "panel_bg":  "#161b22",
}


def render_storage_chart(storage_data: dict, region: str = "europe"):
    import streamlit as st

    df = storage_data.get(region, pd.DataFrame())
    fetched_at = storage_data.get("fetched_at")

    label = "Europe" if region == "europe" else "Germany"
    st.subheader(f"Gas Storage: {label}")

    if df.empty:
        st.warning(
            f"Gas storage data unavailable for {label}. "
            "Add AGSI_API_KEY to .env (free registration at agsi.gie.eu)."
        )
        st.caption("Source: Gas Infrastructure Europe (GIE), AGSI+ | agsi.gie.eu")
        return

    current_year = datetime.utcnow().year
    df_current = df[df["gasDayStart"].dt.year == current_year].copy()
    df_current["doy"] = df_current["gasDayStart"].dt.dayofyear

    bands = build_seasonal_bands(df, value_col="full")

    if bands.empty:
        st.warning(
            "Historical seasonal bands unavailable. The AGSI+ API returned partial data "
            "(likely a timeout on a slow page). Current fill level may still be shown. "
            "Refresh the page to retry."
        )
        return

    # Map day-of-year to calendar dates using a reference year
    ref_year = 2023
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

    # EU mandate reference line at 90% (extended regulation, target by Nov 1)
    fig.add_hline(
        y=90,
        line_dash="dash",
        line_color=COLORS["mandate"],
        line_width=1.2,
        annotation_text="EU mandate (90%, Nov 1)",
        annotation_position="top right",
        annotation_font=dict(size=10, color=COLORS["mandate"]),
    )

    fig.update_layout(
        template="plotly_dark",
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
                f"Storage at {current_pct:.1f}%, below the 5-year minimum for this date. "
                "Critically low heading into the summer refill season."
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
