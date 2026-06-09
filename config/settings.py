"""
Central config: ENTSO-E EIC codes, alert thresholds, zone lists.
Import from here rather than hardcoding in data fetchers or components.
"""

# ── ENTSO-E bidding zone EIC codes ─────────────────────────────────────────
BIDDING_ZONES = {
    "NO1": "10YNO-1--------2",
    "NO2": "10YNO-2--------T",
    "NO3": "10YNO-3--------J",
    "NO4": "10YNO-4--------9",
    "NO5": "10Y1001A1001A48H",
    "SE1": "10Y1001A1001A44P",
    "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L",
    "SE4": "10Y1001A1001A47J",
    "FI":  "10YFI-1--------U",
    "DK1": "10YDK-1--------W",
    "DK2": "10YDK-2--------M",
    "DE":  "10Y1001A1001A83F",
    "NL":  "10YNL----------L",
    "GB":  "10YGB----------A",
    "FR":  "10YFR-RTE------C",
    "IT":  "10YIT-GRTN-----B",
    "ES":  "10YES-REE------0",
    "BE":  "10YBE----------2",
    "AT":  "10YAT-APG------L",
    # Norway aggregate (hydro data)
    "NO":  "10YNO-0--------C",
}

# ── Alert thresholds ────────────────────────────────────────────────────────
TTF_SPIKE_PCT        = 10.0   # % single-day move flagged as spike
SPREAD_WARN_EUR      = 25.0   # Nordic–Continental spread (EUR/MWh) to warn
LNG_SENDOUT_DROP_PCT = 15.0   # Week-on-week sendout drop % to alert
STORAGE_WARN_GAP_PP  = 5.0    # pp below 5-year mean to warn
STORAGE_CRIT_GAP_PP  = 15.0   # pp below 5-year mean to critical-warn

# ── Nord Pool spot price zones ──────────────────────────────────────────────
NORDPOOL_SPOT_ZONES = ["NO1", "NO2", "SE3", "DE-LU", "NL", "FI"]

# ── Cross-border flow pairs for Nordic monitor ──────────────────────────────
FLOW_PAIRS = [
    ("NO1", "SE3"),
    ("NO2", "SE3"),
    ("NO2", "DK1"),
    ("NO2", "DE"),
    ("NO2", "NL"),
    ("NO1", "GB"),
]

# ── Interconnector capacity constants (MW, rounded from operator data) ─────
# Sources: Statnett, TenneT, National Grid, Energinet, Svenska Kraftnät.
# Used for utilisation % display in Layer 1 Flows tab.
# Key format matches FLOW_PAIRS arrow notation: "FROM->TO"
INTERCONNECTOR_CAPACITY_MW: dict[str, float] = {
    "NO2->DE":  1400.0,   # NordLink (Statnett / TenneT, commissioned 2020)
    "NO2->NL":   700.0,   # NorNed (Statnett / TenneT, commissioned 2008)
    "NO1->GB":  1400.0,   # NSN / North Sea Link (Statnett / National Grid, commissioned 2021)
    "NO2->DK1": 1700.0,   # Skagerrak cables 1-4 combined (Statnett / Energinet)
    "NO2->SE3":  2095.0,  # Sydvestlänken (Statnett / SvK, approximate thermal limit)
    "NO1->SE3":   500.0,  # NO1-SE3 (approximate, varies seasonally)
}

# ── Scorecard signal thresholds ─────────────────────────────────────────────
SCORE_RISK_PREMIUM_EUR  = 5.0    # ±EUR/MWh boundary for supply-risk premium direction
SCORE_SEASONAL_HI_PCT   = 75.0   # percentile above which TTF is "historically elevated"
SCORE_SEASONAL_LO_PCT   = 25.0   # percentile below which TTF is "historically cheap"

# ── Nordic spread chart reference ────────────────────────────────────────────
SPREAD_CHART_REF_EUR = 20.0   # ±EUR/MWh annotation level on NL−NO2 spread chart

# ── Interconnector utilisation display thresholds ────────────────────────────
INTERCONNECTOR_UTIL_HIGH_PCT = 90.0   # >= red  (constrained)
INTERCONNECTOR_UTIL_MED_PCT  = 70.0   # >= amber (elevated)

# ── Cointegration & regression model parameters ──────────────────────────────
COINT_ENTRY_Z   = 1.0    # |z-score| threshold to signal a spread trade entry
COINT_MIN_OBS   = 100    # minimum observations required for Engle-Granger test
COINT_MIN_MATCH =  60    # minimum overlapping obs (scorecard OLS, storage regression)

# ── Hydro lead/lag analysis ───────────────────────────────────────────────────
HYDRO_MAX_LAG = 21   # maximum cross-correlation lag in days

# ── Forward Curve PCA (D2) ───────────────────────────────────────────────────
PCA_STORAGE_ALPHA   = 0.03   # storage carry sensitivity (curve adjustment per σ of storage z)
PCA_STORAGE_DECAY_TAU = 6.0  # exponential decay constant (months); storage signal fades on far tenors
PCA_ENTRY_Z         = 2.0    # |z-score| threshold to flag a PC2/PC3 trade signal
PCA_LOOKBACK_DAYS   = 730    # panel history window (~2 years of trading days)

# ── Aluminium smelter stress indicator (NEW2) ────────────────────────────────
ALUM_STRESS_ELEVATED     = 40.0   # power cost as % of smelter revenue → ELEVATED
ALUM_STRESS_CRITICAL     = 55.0   # power cost as % of smelter revenue → CRITICAL
ALUM_ELEC_INTENSITY_MWH  = 14.5   # MWh electricity consumed per tonne of aluminium
ALUM_TTF_TO_ELEC_FACTOR  = 2.0    # TTF→electricity proxy (CCGT marginal cost conversion)
ALUM_EURUSD_APPROX       = 1.10   # fixed EUR/USD for revenue conversion (stated assumption)

# ── LNG countries (GIE ALSI) ────────────────────────────────────────────────
LNG_COUNTRIES = {
    "BE": "Belgium (Zeebrugge)",
    "NL": "Netherlands (Gate LNG)",
    "GB": "Great Britain (S. Hook/Dragon)",
    "FR": "France (Dunkirk/Montoir)",
}
