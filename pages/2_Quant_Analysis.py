"""
Layer 2: Quantitative Analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from datetime import date as _date

load_dotenv()

st.set_page_config(
    page_title="Quantitative Analysis",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.helpers import apply_dark_theme, kpi_card, delta_span, commentary
from data.gas_storage import get_storage_data
from data.prices import get_ttf_data
from data.spot_prices import fetch_spot_prices
from models.storage_monte_carlo import compute_monthly_stats, run_monte_carlo
from models.gas_power_regression import prepare_data, run_full_ols
from models.spike_detector import compute_zscores, latest_signals, ALERT_Z, WARNING_Z
from models.ttf_backtest import fetch_ttf_history, compute_seasonal_strategy, compute_strategy_stats

apply_dark_theme()

# ── Load data ────────────────────────────────────────────────────────────────
with st.spinner(""):
    storage = get_storage_data()
    ttf     = get_ttf_data()
    spot_df = fetch_spot_prices(days=150)   # extended history for regression

eu_df = storage["europe"]

current_pct: float | None = None
if not eu_df.empty and "full" in eu_df.columns:
    current_pct = float(eu_df["full"].iloc[-1])

monthly_stats = compute_monthly_stats(eu_df)
has_empirical = bool(monthly_stats)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## Layer 2: Quantitative Analysis")
st.caption("Quantitative models for European power and gas market analysis.")
st.divider()

with st.expander("Model overview", expanded=False):
    st.markdown("""
    | Model | Status |
    |-------|--------|
    | Storage Refill Monte Carlo | Active |
    | Gas-to-Power OLS Regression (NL proxy) | Active |
    | Price Spike Detector | Active |
    | TTF Seasonal Injection-Withdrawal Backtest | Active |
    | Nordic price decomposition | Awaiting ENTSO-E key (hydro data) |
    | Wind forecast error | Awaiting ENTSO-E key |
    | Merit order / supply stack | In development |
    """)

st.divider()

# ── Model tabs ───────────────────────────────────────────────────────────────
tab_mc, tab_reg, tab_spike, tab_bt = st.tabs([
    "Storage Monte Carlo",
    "Gas-to-Power Regression",
    "Price Spike Detector",
    "TTF Seasonal Strategy",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: STORAGE REFILL MONTE CARLO
# ════════════════════════════════════════════════════════════════════════════
with tab_mc:
    st.markdown("### Storage Refill Monte Carlo Simulation")
    st.caption(
        "Empirical bootstrap simulation of 1,000 storage fill paths from the current "
        "EU aggregate level to November 1. Daily injection rates are sampled from the "
        "historical AGSI distribution by calendar month."
    )

    if current_pct is None:
        st.warning(
            "EU gas storage data unavailable. Add AGSI_API_KEY to .env to enable this model. "
            "Register free at agsi.gie.eu."
        )
    else:
        col_ctrl, col_spacer = st.columns([1, 3])
        with col_ctrl:
            rate_multiplier = st.slider(
                "Injection rate multiplier",
                min_value=0.4,
                max_value=1.5,
                value=1.0,
                step=0.05,
                help=(
                    "1.0x = historical average pace. "
                    "0.6x simulates a slow refill season (supply disruption, low LNG). "
                    "1.3x simulates an accelerated injection scenario."
                ),
            )

        paths, sim_dates = run_monte_carlo(
            current_pct=current_pct,
            monthly_stats=monthly_stats,
            n_paths=1000,
            rate_multiplier=rate_multiplier,
        )

        terminal    = paths[:, -1]
        p_reach_80  = float(np.mean(terminal >= 80.0)) * 100
        p_reach_90  = float(np.mean(terminal >= 90.0)) * 100
        p50_end     = float(np.percentile(terminal, 50))
        pct_dates   = pd.to_datetime(sim_dates)

        p10 = np.percentile(paths, 10, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        p90 = np.percentile(paths, 90, axis=0)

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            color = "green" if p_reach_90 >= 70 else ("amber" if p_reach_90 >= 40 else "red")
            st.markdown(
                kpi_card("Prob(reach 90%, EU mandate)", f"{p_reach_90:.0f}%",
                         delta_span(f"at {rate_multiplier:.2f}x injection rate", color)),
                unsafe_allow_html=True,
            )
        with k2:
            color = "green" if p_reach_80 >= 80 else ("amber" if p_reach_80 >= 55 else "red")
            st.markdown(
                kpi_card("Prob(reach 80%, min threshold)", f"{p_reach_80:.0f}%",
                         delta_span("original 2022 emergency target", color)),
                unsafe_allow_html=True,
            )
        with k3:
            color = "green" if p50_end >= 90 else ("amber" if p50_end >= 75 else "red")
            st.markdown(
                kpi_card("P50 level on Nov 1", f"{p50_end:.1f}%",
                         delta_span("median scenario", color)),
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                kpi_card("Current EU Storage", f"{current_pct:.1f}%",
                         delta_span("current fill level", "blue")),
                unsafe_allow_html=True,
            )

        # Fan chart
        C = {
            "bg": "#0d1117", "panel_bg": "#161b22",
            "grid": "rgba(255,255,255,0.06)", "text": "#c9d1d9",
            "band_out": "rgba(88,166,255,0.12)", "band_in": "rgba(88,166,255,0.27)",
            "median": "#58a6ff", "hist": "#8b949e",
            "mandate": "#d4ac3a", "target": "#3fb950",
        }
        fig = go.Figure()

        if not eu_df.empty and "gasDayStart" in eu_df.columns:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            hp = eu_df[eu_df["gasDayStart"] >= cutoff]
            fig.add_trace(go.Scatter(
                x=hp["gasDayStart"], y=hp["full"],
                name="EU storage (actual)",
                line=dict(color=C["hist"], width=1.8),
                hovertemplate="Actual: %{y:.1f}%<extra></extra>",
            ))

        fig.add_trace(go.Scatter(x=pct_dates, y=p90, line=dict(color="rgba(0,0,0,0)", width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pct_dates, y=p10, name="P10-P90 range",
                                 fill="tonexty", fillcolor=C["band_out"],
                                 line=dict(color="rgba(0,0,0,0)", width=0), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pct_dates, y=p75, line=dict(color="rgba(0,0,0,0)", width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pct_dates, y=p25, name="P25-P75 range",
                                 fill="tonexty", fillcolor=C["band_in"],
                                 line=dict(color="rgba(0,0,0,0)", width=0), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pct_dates, y=p50, name="P50 (median)",
                                 line=dict(color=C["median"], width=2.2),
                                 hovertemplate="P50: %{y:.1f}%<extra></extra>"))
        fig.add_hline(y=90, line_dash="dash", line_color=C["target"], line_width=1.5,
                      annotation_text="90% EU mandate (Nov 1)", annotation_position="right",
                      annotation_font=dict(color=C["target"], size=11))
        fig.add_hline(y=80, line_dash="dot", line_color=C["mandate"], line_width=1.0,
                      annotation_text="80% (2022 original)", annotation_position="right",
                      annotation_font=dict(color=C["mandate"], size=10))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
            font=dict(color=C["text"], size=12),
            margin=dict(l=10, r=90, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor=C["grid"], title=None),
            yaxis=dict(title="Storage fill (%)", showgrid=True, gridcolor=C["grid"],
                       range=[max(0, current_pct - 5), 102]),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        if not has_empirical:
            st.info(
                "Simulation is using fallback injection rates. "
                "Add AGSI_API_KEY to .env for empirically calibrated results."
            )

        with st.expander("Methodology", expanded=True):
            n_hist = int((~eu_df["gasDayStart"].isna()).sum()) if not eu_df.empty else 0
            st.markdown(f"""
            **Method:** Empirical bootstrap Monte Carlo with 1,000 paths.

            **Injection rate sampling:** For each simulated calendar day, a daily injection increment
            (percentage points of working gas volume per day) is drawn with replacement from the
            empirical distribution observed in that calendar month, excluding the current year
            to avoid look-ahead bias. Calibrated from {n_hist:,} AGSI EU aggregate observations
            (approximately 5 years of history).

            **Injection season:** April 1 through October 31. Storage is held flat outside this window.

            **Rate multiplier:** Scales all sampled daily increments uniformly. Allows stress-testing
            of below-average (supply disruption, low LNG diversion) or above-average injection seasons.
            The random seed is fixed per multiplier value for reproducible scenario comparisons.

            **Reference levels:**
            - **90% (green dashed):** Current EU mandate. The original 2022 emergency regulation (EU 2022/1032)
              set 80%, but subsequent extensions raised the target to 90% by November 1 for the 2023–2025+
              period. In 2025, EU storage reached 83% by October 1 and met the 90% mandate by November 1.
            - **80% (gold dotted):** The original 2022 emergency target under EU Regulation 2022/1032.
              Shown for historical reference.

            **Percentile bands:** Outer band (P10-P90) contains 80% of paths. Inner band (P25-P75) is the
            interquartile range. The blue line is the P50 median path.

            **Limitations:** Model assumes injection rates are IID within a given month across years.
            Does not capture serial correlation, geopolitical supply disruptions, or demand-side shocks.
            Use alongside live market signals on the Monitor page.
            """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: GAS-TO-POWER REGRESSION
# ════════════════════════════════════════════════════════════════════════════
with tab_reg:
    st.markdown("### Gas-to-Power OLS Regression")
    st.caption(
        "OLS regression of Netherlands day-ahead electricity price on TTF front-month gas price. "
        "NL is the most direct pairing with TTF (both Netherlands-based markets). "
        "Residuals identify dates where power is priced above or below what gas fundamentals imply."
    )

    POWER_ZONE = "NL"
    ttf_df = ttf["prices"]
    reg_df = prepare_data(ttf_df, spot_df, power_zone=POWER_ZONE)
    ols    = run_full_ols(reg_df)

    if not ols:
        st.warning(
            "Insufficient data for regression. Requires at least 20 days of overlapping "
            "TTF and NL day-ahead prices. Spot price data may still be loading."
        )
    else:
        n_obs       = ols["n_obs"]
        slope       = ols["slope"]
        intercept   = ols["intercept"]
        r2          = ols["r2"]
        residuals   = ols["residual"]
        z_scores    = ols["residual_zscore"]
        fitted_arr  = ols["fitted"]

        latest_z     = float(z_scores[-1])
        latest_resid = float(residuals[-1])
        latest_act   = float(reg_df["power_price"].iloc[-1])
        latest_fit   = float(fitted_arr[-1])

        z_color  = "red" if abs(latest_z) > 2 else ("amber" if abs(latest_z) > 1.5 else "green")
        # Low R² is expected in a renewables-dominated regime; signal value is in residuals, not fit
        r2_regime = "renewables-dominated (residual signal valid)" if r2 < 0.3 else (
            "moderate gas-power coupling" if r2 < 0.6 else "strong gas-marginal regime"
        )
        r2_color = "amber"  # neutral: low R² is regime information, not a model failure

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                kpi_card("Residual z-score (latest)", f"{latest_z:+.2f}",
                         delta_span("above" if latest_z > 0 else "below gas-model price", z_color)),
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                kpi_card("R-squared (full sample)", f"{r2:.2f}",
                         delta_span(r2_regime, r2_color)),
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                kpi_card("Beta (TTF to NL power)", f"{slope:.2f}",
                         delta_span("EUR/MWh power per EUR/MWh gas", "blue")),
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                kpi_card("Residual (actual - fitted)", f"{latest_resid:+.1f} EUR/MWh",
                         delta_span(f"actual {latest_act:.1f}, model {latest_fit:.1f}", z_color)),
                unsafe_allow_html=True,
            )

        C = {
            "bg": "#0d1117", "panel_bg": "#161b22",
            "grid": "rgba(255,255,255,0.06)", "text": "#c9d1d9",
            "scatter": "#58a6ff", "regline": "#d4ac3a",
            "actual": "#c9d1d9", "fitted": "#58a6ff",
            "resid_pos": "#f85149", "resid_neg": "#3fb950",
            "zero": "rgba(255,255,255,0.3)",
        }

        # Two-column layout: scatter (left), residuals time series (right)
        col_scat, col_resid = st.columns([1, 1])

        # Scatter plot
        with col_scat:
            x_range = [float(reg_df["ttf_price"].min()) * 0.97,
                       float(reg_df["ttf_price"].max()) * 1.03]
            y_line  = [slope * x + intercept for x in x_range]

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=reg_df["ttf_price"], y=reg_df["power_price"],
                mode="markers",
                marker=dict(color=C["scatter"], size=5, opacity=0.7),
                name="Daily observations",
                hovertemplate="TTF: €%{x:.1f}, NL: €%{y:.1f}<extra></extra>",
            ))
            # Latest point highlighted
            fig_sc.add_trace(go.Scatter(
                x=[float(reg_df["ttf_price"].iloc[-1])],
                y=[latest_act],
                mode="markers",
                marker=dict(color="#f85149", size=9, symbol="circle",
                            line=dict(color="white", width=1)),
                name="Latest",
                hovertemplate=f"Latest: TTF €{reg_df['ttf_price'].iloc[-1]:.1f}, NL €{latest_act:.1f}<extra></extra>",
            ))
            fig_sc.add_trace(go.Scatter(
                x=x_range, y=y_line,
                mode="lines",
                line=dict(color=C["regline"], width=1.5, dash="dot"),
                name=f"OLS fit (R²={r2:.2f}, β={slope:.2f})",
                hoverinfo="skip",
            ))
            fig_sc.update_layout(
                title=dict(text="NL Power vs. TTF Gas Price (EUR/MWh)", font=dict(size=12)),
                template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
                font=dict(color=C["text"], size=11),
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="TTF (EUR/MWh)", showgrid=True, gridcolor=C["grid"]),
                yaxis=dict(title="NL day-ahead (EUR/MWh)", showgrid=True, gridcolor=C["grid"]),
                height=340,
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # Residuals time series
        with col_resid:
            z_vals  = z_scores
            z_dates = reg_df["date"].values
            colors  = [C["resid_pos"] if z > 0 else C["resid_neg"] for z in z_vals]

            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(
                x=z_dates, y=z_vals,
                marker_color=colors,
                name="Residual z-score",
                hovertemplate="z: %{y:.2f}<extra></extra>",
            ))
            for level, label in [(2.0, "+2\u03c3"), (-2.0, "\u22122\u03c3")]:
                fig_res.add_hline(y=level, line_dash="dash",
                                  line_color="rgba(248,81,73,0.55)", line_width=1)
            fig_res.add_hline(y=0, line_dash="solid",
                              line_color=C["zero"], line_width=0.8)
            fig_res.update_layout(
                title=dict(text="Standardised Residual (z-score)", font=dict(size=12)),
                template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
                font=dict(color=C["text"], size=11),
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(showgrid=True, gridcolor=C["grid"], title=None),
                yaxis=dict(showgrid=True, gridcolor=C["grid"], title="z-score"),
                showlegend=False,
                hovermode="x unified",
                height=340,
            )
            st.plotly_chart(fig_res, use_container_width=True)

        # Signal interpretation
        if abs(latest_z) >= 2.0:
            direction = "above" if latest_z > 0 else "below"
            implication = (
                "NL power is unusually expensive relative to the gas cost curve. "
                "Possible drivers: North Sea wind drought, cold-snap demand spike, or interconnector constraints. "
                if latest_z > 0 else
                "NL power is unusually cheap relative to the gas cost curve. "
                "Possible drivers: North Sea wind surplus, weak industrial demand, "
                "or strong NorNed imports from Norway."
            )
            st.markdown(
                commentary(
                    f"Signal: NL day-ahead at {latest_act:.1f} EUR/MWh is {abs(latest_resid):.1f} EUR/MWh "
                    f"{direction} the TTF gas-model prediction ({latest_fit:.1f} EUR/MWh), "
                    f"a {abs(latest_z):.1f} standard deviation residual over the {n_obs}-day sample. "
                    + implication,
                    "critical" if abs(latest_z) > 2.5 else "warn",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                f"Latest NL day-ahead ({latest_act:.1f} EUR/MWh) is within normal range relative to "
                f"the TTF gas-model ({latest_fit:.1f} EUR/MWh). Residual z-score: {latest_z:+.2f}."
            )

        with st.expander("Methodology", expanded=True):
            st.markdown(f"""
            **Model:** OLS regression of Netherlands day-ahead electricity price on TTF front-month gas
            price, estimated over the full available sample of {n_obs} overlapping trading days.

            **Estimated coefficients:** Intercept {intercept:.1f} EUR/MWh, slope {slope:.2f}
            (EUR/MWh power per EUR/MWh gas), R-squared {r2:.2f}.

            **Why R²={r2:.2f} is not a model failure:**
            In a gas-marginal market (2021–2022), gas explained ~60–70% of power price variation.
            The current sample period reflects a renewables-dominated regime: on many days the marginal
            generator is wind or solar (near-zero marginal cost), so TTF has weak explanatory power
            for the absolute price level. R²={r2:.2f} confirms this: it is regime information, not a
            coding error. The model's value is not prediction accuracy; it is the **residual z-score**:
            on days when gas should explain power but the residual is extreme (|z|>2), there is an
            identifiable fundamental dislocation (wind drought, demand shock, interconnector constraint).

            **Data sources:**
            - NL day-ahead price: Nord Pool Data Portal (daily average, EUR/MWh)
            - TTF price: ICE TTF front-month futures via Yahoo Finance (EUR/MWh, approximately 120 calendar days / ~85 trading days)

            **Why NL rather than DE:** Nord Pool's public data portal returns null prices for DE-LU, which
            falls under separate EPEX Spot data licensing. NL is an appropriate proxy: it is directly
            connected to Germany and Belgium and is priced within the same Central West European (CWE)
            market coupling zone. The TTF gas hub is located in the Netherlands, making the NL power-gas
            relationship especially direct.

            **Residual z-score:** The raw residual (actual minus fitted, EUR/MWh) is normalised by the
            standard deviation of residuals over the sample. A z-score exceeding +2 indicates NL power
            is materially expensive relative to what the gas price implies; below -2 indicates cheapness.
            The z-score is the primary signal; the R² is secondary context.

            **Interpretation of beta ({slope:.2f}):**
            In a gas-marginal power market, the theoretical beta equals 1 divided by the CCGT heat rate
            efficiency (approximately 0.55-0.60), implying beta in the range 1.6-1.8. A beta below this
            range confirms renewables are partially displacing gas at the margin in the current sample.

            **Limitations:** The model is bivariate and omits coal price, carbon (EU ETS) price,
            wind and solar output, and hydro availability. It identifies deviations from the historical
            gas-power relationship but does not attribute them to specific fundamental drivers.
            """)

        st.caption("Sources: Nord Pool Data Portal (NL day-ahead) | ICE/Yahoo Finance (TTF)")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: PRICE SPIKE DETECTOR
# ════════════════════════════════════════════════════════════════════════════
with tab_spike:
    st.markdown("### Day-Ahead Price Spike Detector")
    st.caption(
        f"Rolling 30-day z-score on daily day-ahead prices for each bidding zone. "
        f"Alert threshold: |z| > {ALERT_Z}. Warning threshold: |z| > {WARNING_Z}. "
        "Identifies zones with anomalous price levels relative to recent history."
    )

    spike_df  = spot_df   # already fetched at page load (days=150, more than sufficient)
    z_df      = compute_zscores(spike_df, window=30)
    latest_z  = latest_signals(z_df)

    if z_df.empty:
        st.warning("Spot price data unavailable for spike detection.")
    else:
        # Alert summary KPIs
        n_alert = int((latest_z["signal"] == "alert").sum()) if not latest_z.empty else 0
        n_warn  = int((latest_z["signal"] == "warn").sum())  if not latest_z.empty else 0
        n_norm  = int((latest_z["signal"] == "normal").sum()) if not latest_z.empty else 0

        ka, kw, kn, kd = st.columns(4)
        with ka:
            color = "red" if n_alert > 0 else "green"
            st.markdown(kpi_card("Zones in alert", str(n_alert),
                                 delta_span(f"|z| > {ALERT_Z}", color)), unsafe_allow_html=True)
        with kw:
            color = "amber" if n_warn > 0 else "green"
            st.markdown(kpi_card("Zones in warning", str(n_warn),
                                 delta_span(f"|z| > {WARNING_Z}", color)), unsafe_allow_html=True)
        with kn:
            st.markdown(kpi_card("Zones normal", str(n_norm),
                                 delta_span("within range", "green")), unsafe_allow_html=True)
        with kd:
            latest_date = z_df["date"].max() if not z_df.empty else "n/a"
            st.markdown(kpi_card("Latest data", str(latest_date),
                                 delta_span("30-day rolling window", "blue")), unsafe_allow_html=True)

        # Zone signal table
        if not latest_z.empty:
            st.markdown("#### Current Signal by Zone")
            ZONE_LABELS = {
                "NO1": "NO1 (Oslo)", "NO2": "NO2 (Kristiansand)",
                "SE3": "SE3 (Stockholm)", "NL": "NL (Netherlands)", "FI": "FI (Finland)",
            }
            SIG_PILL = {
                "alert":  '<span class="pill-critical">ALERT</span>',
                "warn":   '<span class="pill-warn">WARN</span>',
                "normal": '<span class="pill-ok">NORMAL</span>',
            }
            for _, row in latest_z.iterrows():
                zone   = row["zone"]
                price  = row["price_eur_mwh"]
                z      = row["z_score"]
                mean   = row["rolling_mean"]
                sig    = row["signal"]
                label  = ZONE_LABELS.get(zone, zone)
                pill   = SIG_PILL.get(sig, "")
                z_str  = f"{z:+.2f}" if not pd.isna(z) else "n/a"
                mean_s = f"{mean:.1f}" if not pd.isna(mean) else "n/a"
                st.markdown(
                    f"{pill} &nbsp; <strong>{label}</strong> &nbsp; "
                    f"€{price:.1f}/MWh &nbsp; z = {z_str} &nbsp; "
                    f"<span style='color:#8b949e;font-size:0.82rem;'>(30d avg: €{mean_s})</span>",
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

        # Z-score time series chart
        C = {
            "bg": "#0d1117", "panel_bg": "#161b22",
            "grid": "rgba(255,255,255,0.06)", "text": "#c9d1d9",
        }
        ZONE_COLORS = {
            "NO1": "#4caf8f", "NO2": "#2980b9", "SE3": "#7ec8e3",
            "NL":  "#c0392b", "FI":  "#8e6bbf",
        }

        fig_z = go.Figure()
        for zone in sorted(z_df["zone"].unique()):
            sub = z_df[z_df["zone"] == zone].dropna(subset=["z_score"]).sort_values("date")
            if sub.empty:
                continue
            label = ZONE_LABELS.get(zone, zone)
            fig_z.add_trace(go.Scatter(
                x=sub["date"], y=sub["z_score"],
                name=label,
                line=dict(color=ZONE_COLORS.get(zone, "#888"), width=1.5),
                hovertemplate=f"{label}: z=%{{y:.2f}}<extra></extra>",
            ))

        for level, color, dash in [
            (ALERT_Z,  "rgba(248,81,73,0.7)",  "dash"),
            (-ALERT_Z, "rgba(248,81,73,0.7)",  "dash"),
            (WARNING_Z, "rgba(210,153,34,0.5)", "dot"),
            (-WARNING_Z,"rgba(210,153,34,0.5)", "dot"),
            (0,         "rgba(255,255,255,0.2)", "solid"),
        ]:
            fig_z.add_hline(y=level, line_dash=dash, line_color=color, line_width=1)

        fig_z.update_layout(
            template="plotly_dark",
            paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
            font=dict(color=C["text"], size=12),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor=C["grid"], title=None),
            yaxis=dict(title="z-score (30-day rolling)", showgrid=True, gridcolor=C["grid"]),
            hovermode="x unified",
        )
        st.plotly_chart(fig_z, use_container_width=True)

        with st.expander("Methodology", expanded=False):
            st.markdown(f"""
            **Method:** For each bidding zone, the rolling 30-day mean and standard deviation of daily
            average day-ahead prices are computed. The z-score is defined as:
            `z = (price - rolling_mean) / rolling_std`.

            **Thresholds:**
            - |z| > {ALERT_Z}: Alert. The current price is {ALERT_Z} standard deviations above or below
              the 30-day historical average. This is anomalous relative to recent price history.
            - |z| > {WARNING_Z}: Warning. Elevated deviation; monitor for further moves.
            - |z| < {WARNING_Z}: Normal range.

            **Positive z:** Current price is above recent average. Possible causes: cold snap, wind drought,
            outage, or supply disruption.

            **Negative z:** Current price is below recent average. Possible causes: renewable surplus,
            weak demand, strong hydro inflow, or excess interconnector imports.

            **Limitations:** The z-score is relative to recent history (30 days), not long-run levels.
            A zone can show z = 0 while trading at historically elevated absolute prices if it has been
            consistently high for the past 30 days. Always compare against the absolute price level.

            Data source: Nord Pool Data Portal, day-ahead market prices.
            """)

        st.caption("Source: Nord Pool Data Portal, Day-Ahead Market Prices | nordpoolgroup.com")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4: TTF SEASONAL INJECTION-WITHDRAWAL BACKTEST
# ════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown("### TTF Seasonal Injection-Withdrawal Strategy Backtest")
    st.caption(
        "Rules-based strategy: buy summer TTF (Apr-Sep average), sell winter TTF (Oct-Mar average) "
        "when the seasonal spread exceeds round-trip storage cost. "
        "P&L is expressed per MWh of storage capacity deployed."
    )

    col_ctrl, col_spacer = st.columns([1, 3])
    with col_ctrl:
        storage_cost = st.slider(
            "Storage cost (EUR/MWh round-trip)",
            min_value=1.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            help=(
                "Round-trip cost of holding gas in storage: injection fee + withdrawal fee + "
                "fuel shrinkage, typically €3-5/MWh for European underground storage. "
                "Slide up to stress-test the strategy against higher costs."
            ),
        )

    ttf_hist = fetch_ttf_history(years=7)
    bt = compute_seasonal_strategy(ttf_hist, storage_cost=storage_cost)
    stats = compute_strategy_stats(bt)

    if bt.empty:
        st.warning(
            "TTF historical data unavailable. "
            "Requires yfinance and internet access (ticker: TTF=F)."
        )
    else:
        # ── KPI row ──────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            hit_color = "green" if stats["hit_rate"] >= 60 else ("amber" if stats["hit_rate"] >= 40 else "red")
            st.markdown(
                kpi_card("Hit rate (full sample)", f"{stats['hit_rate']:.0f}%",
                         delta_span(f"profitable in {int(stats['hit_rate']*stats['n_years']/100)}/{stats['n_years']} gas years", hit_color)),
                unsafe_allow_html=True,
            )
        with k2:
            pnl_color = "green" if stats["avg_pnl"] > 0 else "red"
            st.markdown(
                kpi_card("Avg P&L (full sample)", f"€{stats['avg_pnl']:.1f}/MWh",
                         delta_span("includes 2021-22 energy crisis", pnl_color)),
                unsafe_allow_html=True,
            )
        with k3:
            exc = stats.get("avg_pnl_ex_crisis")
            if exc is not None:
                exc_color = "green" if exc > 0 else ("amber" if exc > -2 else "red")
                exc_n     = stats.get("n_ex_crisis", 0)
                st.markdown(
                    kpi_card("Avg P&L (ex-crisis)", f"€{exc:.1f}/MWh",
                             delta_span(f"excl. GY2021-22 · {exc_n} years", exc_color)),
                    unsafe_allow_html=True,
                )
            else:
                sharpe_color = "green" if stats["sharpe"] >= 0.5 else ("amber" if stats["sharpe"] >= 0 else "red")
                st.markdown(
                    kpi_card("Sharpe ratio", f"{stats['sharpe']:.2f}",
                             delta_span("annual P&L / std dev", sharpe_color)),
                    unsafe_allow_html=True,
                )
        with k4:
            st.markdown(
                kpi_card("Avg summer-winter spread", f"€{stats['avg_spread']:.1f}/MWh",
                         delta_span("before storage cost", "blue")),
                unsafe_allow_html=True,
            )

        C = {
            "bg": "#0d1117", "panel_bg": "#161b22",
            "grid": "rgba(255,255,255,0.06)", "text": "#c9d1d9",
            "pos": "#3fb950", "neg": "#f85149",
            "cum": "#58a6ff", "zero": "rgba(255,255,255,0.25)",
            "cost": "rgba(210,180,40,0.55)",
        }

        # ── Annual P&L bar chart ─────────────────────────────────────────────
        bar_colors = [C["pos"] if v >= 0 else C["neg"] for v in bt["pnl"]]

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Bar(
            x=bt["label"],
            y=bt["pnl"],
            marker_color=bar_colors,
            name="Annual P&L",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "P&L: €%{y:.1f}/MWh<br>"
                "<extra></extra>"
            ),
        ))
        fig_bt.add_hline(
            y=0,
            line_dash="solid",
            line_color=C["zero"],
            line_width=0.8,
        )
        fig_bt.update_layout(
            title=dict(text="Annual P&L per MWh (net of storage cost)", font=dict(size=12)),
            template="plotly_dark",
            paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
            font=dict(color=C["text"], size=12),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False, title=None),
            yaxis=dict(title="EUR/MWh", showgrid=True, gridcolor=C["grid"]),
            hovermode="x unified",
            height=300,
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # ── Cumulative P&L line ──────────────────────────────────────────────
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=bt["label"],
            y=bt["cum_pnl"],
            mode="lines+markers",
            line=dict(color=C["cum"], width=2.2),
            marker=dict(size=6, color=C["cum"]),
            name="Cumulative P&L",
            hovertemplate="Cumulative: €%{y:.1f}/MWh<extra></extra>",
        ))
        fig_cum.add_hline(y=0, line_dash="solid", line_color=C["zero"], line_width=0.8)
        fig_cum.update_layout(
            title=dict(text="Cumulative P&L per MWh", font=dict(size=12)),
            template="plotly_dark",
            paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
            font=dict(color=C["text"], size=12),
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            xaxis=dict(showgrid=False, title=None),
            yaxis=dict(title="EUR/MWh", showgrid=True, gridcolor=C["grid"]),
            hovermode="x",
            height=220,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── Crisis context note ──────────────────────────────────────────────
        exc = stats.get("avg_pnl_ex_crisis")
        if exc is not None and stats["n_years"] > 0:
            crisis_rows = bt[bt["gas_year"].astype(int).isin([2021, 2022])]
            if not crisis_rows.empty:
                crisis_pnl_str = ", ".join(
                    f"{row['label']}: €{row['pnl']:.0f}/MWh"
                    for _, row in crisis_rows.iterrows()
                )
                ex_n = stats.get("n_ex_crisis", 0)
                st.caption(
                    f"Note: GY2021 and GY2022 reflect the European energy crisis, with geopolitical-shock spreads "
                    f"that dominate the full-sample average ({crisis_pnl_str}). "
                    f"Excluding these years, the average P&L across {ex_n} normal gas years is "
                    f"€{exc:.1f}/MWh, which is a more representative picture of steady-state storage economics."
                )

        # ── Results table ────────────────────────────────────────────────────
        with st.expander("Per-year results table", expanded=False):
            display = bt[["label", "summer_avg", "winter_avg", "spread", "pnl"]].copy()
            display.columns = ["Gas Year", "Summer avg (€/MWh)", "Winter avg (€/MWh)",
                                "Spread (€/MWh)", f"P&L net of €{storage_cost:.1f} cost (€/MWh)"]
            st.dataframe(display.set_index("Gas Year"), use_container_width=True)

        # ── Signal: current gas year injection economics ──────────────────────
        if not ttf_hist.empty:
            ttf_hist_dated = ttf_hist.copy()
            ttf_hist_dated["date"] = pd.to_datetime(ttf_hist_dated["date"])
            ttf_hist_dated["month"] = ttf_hist_dated["date"].dt.month
            ttf_hist_dated["year"]  = ttf_hist_dated["date"].dt.year

            today_ts = pd.Timestamp.now()
            # Gas year label = calendar year of summer half (Apr onwards)
            gy_start = today_ts.year if today_ts.month >= 4 else today_ts.year - 1

            curr_summer = ttf_hist_dated[
                (ttf_hist_dated["year"] == gy_start) &
                (ttf_hist_dated["month"].between(4, 9))
            ]
            curr_price = float(ttf_hist_dated["price"].iloc[-1])

            if not curr_summer.empty:
                curr_summer_avg = float(curr_summer["price"].mean())
                n_summer_days   = len(curr_summer)
                breakeven_winter = round(curr_summer_avg + storage_cost, 1)
                st.markdown(
                    commentary(
                        f"GY {gy_start}/{str(gy_start + 1)[-2:]}: summer injection season underway "
                        f"({n_summer_days} trading days, average summer price so far: "
                        f"€{curr_summer_avg:.1f}/MWh). "
                        f"Break-even winter average (summer avg + €{storage_cost:.1f}/MWh storage cost): "
                        f"€{breakeven_winter:.1f}/MWh. "
                        f"Winter forwards above this level imply a profitable storage trade.",
                        "ok",
                    ),
                    unsafe_allow_html=True,
                )

        with st.expander("Strategy methodology", expanded=True):
            st.markdown(f"""
            **Strategy:** Seasonal carry trade modelled from the perspective of a physical storage
            operator (always-in). For each gas year, gas is injected at summer prices (Apr-Sep) and
            withdrawn at winter prices (Oct-Mar). The net P&L per MWh is the realised
            winter-minus-summer average price spread, minus the round-trip cost of storage.
            A financial trader would only enter when the spread exceeds storage cost; a physical
            operator runs the position every year regardless.

            **Gas year definition:** April 1 of year Y to March 31 of year Y+1.
            - Summer half: April–September (injection season)
            - Winter half: October–March (withdrawal season)

            **P&L calculation:**
            `P&L = mean(Winter prices) − mean(Summer prices) − storage_cost`

            **Storage cost (€{storage_cost:.1f}/MWh):** Set via the slider above.
            Round-trip European underground storage typically costs €3-5/MWh, covering injection
            and withdrawal tariffs, fuel shrinkage, and financing costs. Higher for LNG re-gassing.

            **Proxy:** TTF front-month continuous futures (Yahoo Finance TTF=F). The front-month
            price is used as a proxy for the average injection/withdrawal price. A more precise
            backtest would use the specific summer and winter futures contracts (e.g., ICE TTF
            Q3 and Q1/Q2 strips), but front-month provides a directionally accurate proxy.

            **Sharpe ratio:** Computed as mean(annual P&L) / std(annual P&L), treating each
            gas year as one independent observation. Not annualised further.

            **Limitations:** Does not model:
            - Cost of capital (margin requirements on ICE TTF futures)
            - Basis risk between front-month and actual strip prices
            - Storage capacity constraints or fill-level path dependency
            - Transaction costs beyond the storage cost assumption
            Use as a directional indicator of seasonal spread dynamics, not as a precise
            trading P&L estimate.

            **Data:** ICE TTF front-month continuous futures via Yahoo Finance (TTF=F),
            approximately {len(ttf_hist):,} trading days of history.
            """)

        st.caption("Source: ICE/Yahoo Finance (TTF=F) | agsi.gie.eu")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """<div style="color:#484f58;font-size:0.72rem;line-height:1.8;">
    Built by Tobias Gudbjerg &nbsp;|&nbsp;
    Storage data: <a href="https://agsi.gie.eu" style="color:#484f58;">GIE AGSI+</a> &nbsp;|&nbsp;
    Gas price: <a href="https://finance.yahoo.com" style="color:#484f58;">ICE/Yahoo Finance (TTF)</a> &nbsp;|&nbsp;
    Power price: <a href="https://www.nordpoolgroup.com" style="color:#484f58;">Nord Pool</a><br>
    For informational purposes only. Not financial advice.
    </div>""",
    unsafe_allow_html=True,
)
