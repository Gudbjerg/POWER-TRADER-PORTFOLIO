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

# ── LNG countries (GIE ALSI) ────────────────────────────────────────────────
LNG_COUNTRIES = {
    "BE": "Belgium (Zeebrugge)",
    "NL": "Netherlands (Gate LNG)",
    "GB": "Great Britain (S. Hook/Dragon)",
    "FR": "France (Dunkirk/Montoir)",
}
