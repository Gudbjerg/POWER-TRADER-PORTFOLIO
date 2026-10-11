"""
Scenario interpretation logic. Translates raw data into trader-readable signals.
Returns (status, headline, detail) tuples where status is one of: critical, warn, ok, good.
"""
from __future__ import annotations
from datetime import date
import pandas as pd

# EU storage regulation target used in the commentary: 90% full by 1 November.
STORAGE_TARGET_PCT = 90.0


# ── Gas Storage ─────────────────────────────────────────────────────────────

def _injection_season(today: date) -> bool:
    """True from 1 April to 31 October, the usual net-injection season."""
    return 4 <= today.month <= 10


def storage_status(
    pct: float,
    min_pct: float | None,
    mean_pct: float | None,
    region: str = "EU",
    today: date | None = None,
) -> tuple[str, str, str]:
    if mean_pct is None or min_pct is None:
        return ("ok", f"{region} storage at {pct:.1f}%; historical bands unavailable", "")
    today = today or date.today()
    injecting = _injection_season(today)
    gap_to_mean = pct - mean_pct
    gap_to_min  = pct - min_pct

    if pct < min_pct:
        return (
            "critical",
            f"{region} storage at historic low: {pct:.1f}% (below 5-year minimum of {min_pct:.1f}%)",
            (
                f"Storage has fallen below every prior year's level for this date over the last five years. "
                + (
                    f"At {pct:.1f}%, the market faces a steep injection requirement before the winter "
                    f"withdrawal season. "
                    if injecting else
                    f"At {pct:.1f}%, stocks may run short before the end of the withdrawal season, and "
                    f"next summer's refill requirement grows with every week of deficit. "
                )
                + f"A cold snap or LNG supply disruption could rapidly tighten balances. "
                f"Watch the EU storage target ({STORAGE_TARGET_PCT:.0f}% by 1 November). The gap to target is significant."
            ),
        )
    elif gap_to_mean < -15:
        return (
            "warn",
            f"{region} storage critically below the seasonal average: {pct:.1f}% versus {mean_pct:.1f}%",
            (
                f"Storage is {abs(gap_to_mean):.1f}pp below the 5-year average for this date. "
                f"While above the historical minimum, the pace of injection required to reach "
                f"{STORAGE_TARGET_PCT:.0f}% by 1 November is well above seasonal norms. "
                + (
                    "Higher TTF prices are likely to incentivise LNG cargoes and pipeline volumes."
                    if injecting else
                    "Withdrawals from a low base raise the risk premium on late-winter delivery."
                )
            ),
        )
    elif gap_to_mean < -5:
        return (
            "warn",
            f"{region} storage below the seasonal average: {pct:.1f}% ({abs(gap_to_mean):.1f}pp below 5-year mean)",
            (
                f"Storage is running {abs(gap_to_mean):.1f}pp below the 5-year seasonal average. "
                f"Injection pace will need to exceed historical norms to reach winter targets. "
                f"Near-term TTF prices are supported by refill demand."
            ),
        )
    elif gap_to_mean > 10:
        return (
            "good",
            f"{region} storage well above the seasonal average: {pct:.1f}% ({gap_to_mean:+.1f}pp above 5-year mean)",
            (
                f"Stocks are comfortably above the seasonal average, reducing near-term supply risk. "
                f"This is bearish for TTF in the short term unless winter temperatures are extreme."
            ),
        )
    else:
        return (
            "ok",
            f"{region} storage near the seasonal average: {pct:.1f}% ({gap_to_mean:+.1f}pp versus 5-year mean)",
            f"Storage is tracking close to the 5-year seasonal average. No acute supply risk.",
        )


def storage_refill_pace(
    pct: float,
    bands: pd.DataFrame,
    today: date,
    target_pct: float = STORAGE_TARGET_PCT,
) -> dict:
    """Compare the injection still needed to reach the 1 November target with the 5-year norm.

    ``bands`` is the output of ``data.gas_storage.build_seasonal_bands`` (day_of_year,
    min, mean, max). The 5-year norm for the remaining period is the mean fill on
    1 November minus the mean fill on today's day of year: the net injection that the
    average of the last five years achieved over the same calendar window.

    Returns a dict with ``days`` (to the next 1 November), ``required_pp``,
    ``required_per_day``, ``norm_pp`` and ``norm_per_day`` (None when the bands cannot
    supply them) and ``verdict``: one of "met", "well above", "above", "in line with",
    "below" or None.
    """
    nov1 = date(today.year if today < date(today.year, 11, 1) else today.year + 1, 11, 1)
    days = max((nov1 - today).days, 1)
    required = max(0.0, target_pct - float(pct))
    out = {
        "days": days,
        "required_pp": required,
        "required_per_day": required / days,
        "norm_pp": None,
        "norm_per_day": None,
        "verdict": "met" if required <= 0 else None,
    }
    if required <= 0 or bands is None or bands.empty:
        return out
    by_doy = bands.set_index("day_of_year")["mean"]
    today_doy = pd.Timestamp(today).dayofyear
    nov1_doy = pd.Timestamp(nov1).dayofyear
    if today_doy not in by_doy.index or nov1_doy not in by_doy.index:
        return out
    norm = float(by_doy.loc[nov1_doy]) - float(by_doy.loc[today_doy])
    out["norm_pp"] = norm
    out["norm_per_day"] = norm / days
    if norm <= 0.1:
        out["verdict"] = "well above"
    else:
        ratio = required / norm
        out["verdict"] = (
            "well above" if ratio >= 1.5 else
            "above" if ratio >= 1.1 else
            "in line with" if ratio >= 0.9 else
            "below"
        )
    return out


# ── TTF Gas Price ────────────────────────────────────────────────────────────

def ttf_status(price: float, ma30: float, ma90: float, spike: bool, spike_pct: float) -> tuple[str, str, str]:
    if spike:
        direction = "surged" if spike_pct > 0 else "collapsed"
        return (
            "critical",
            f"TTF spike: price {direction} {abs(spike_pct):.1f}% intraday to €{price:.2f}/MWh",
            (
                f"A move of this magnitude typically reflects an acute supply event "
                f"(pipeline disruption, LNG facility outage, or unexpected cold snap). "
                f"Check for news on Norwegian pipeline flows, Algerian supply, and weather forecasts."
            ),
        )
    elif price > 50:
        return (
            "warn",
            f"TTF elevated at €{price:.2f}/MWh ({(price/ma30-1)*100:+.1f}% versus 30-day average)",
            (
                f"TTF above €50/MWh reflects tight storage fundamentals and supply uncertainty. "
                f"At these levels, industrial gas demand destruction becomes a factor. "
                f"Power sector switching from gas to coal increases where coal plant capacity remains available."
            ),
        )
    elif price > 35:
        return (
            "warn",
            f"TTF above long-run average at €{price:.2f}/MWh",
            (
                f"Prices are running above the pre-crisis long-run average (approximately €20/MWh) "
                f"but well below 2022 crisis highs. Low storage is the primary support factor. "
                f"Summer refill demand and LNG import capacity are the key variables to monitor."
            ),
        )
    elif price < 20:
        return (
            "good",
            f"TTF at €{price:.2f}/MWh, providing strong injection economics",
            (
                f"Low gas prices incentivise aggressive summer injection. "
                f"If sustained, storage should recover toward the seasonal average. "
                f"Bearish signal for power prices in gas-heavy generation mixes."
            ),
        )
    else:
        return (
            "ok",
            f"TTF within normal range: €{price:.2f}/MWh ({(price/ma30-1)*100:+.1f}% versus 30-day average)",
            f"Gas prices within the post-crisis normal range. No acute signal.",
        )


# ── Nordic–Continental Price Spread ─────────────────────────────────────────

def spread_status(nordic_avg: float, continental_avg: float) -> tuple[str, str, str]:
    spread = continental_avg - nordic_avg   # positive = Continent more expensive

    if spread > 25:
        return (
            "warn",
            f"Wide Nordic-NL spread: NL €{spread:.0f}/MWh above Nordic (DE-LU unavailable, NL used as proxy)",
            (
                f"A spread of this magnitude implies interconnectors (NordLink, NorNed, NSN, NordBalt) "
                f"are likely operating at or near full capacity. "
                f"High Continental prices reflect gas and coal generation costs. "
                f"Strong incentive for Norwegian hydro export. Watch reservoir levels."
            ),
        )
    elif spread > 10:
        return (
            "ok",
            f"Moderate export premium: NL €{spread:.0f}/MWh above Nordic",
            (
                f"NL prices are materially above Nordic, providing a clear export incentive "
                f"for Norwegian hydro. Flows are likely running at high utilisation on key cables."
            ),
        )
    elif spread < -10:
        return (
            "warn",
            f"Reversed spread: Nordic prices €{abs(spread):.0f}/MWh above NL",
            (
                f"Nordic prices above NL is atypical and may signal hydro scarcity "
                f"(low reservoir levels), a cable outage, or unusually high Nordic demand. "
                f"Check Norwegian reservoir data and interconnector availability."
            ),
        )
    else:
        return (
            "ok",
            f"Nordic-NL spread narrow at €{spread:+.0f}/MWh",
            f"Markets are broadly coupled. Normal flow conditions on interconnectors.",
        )


# ── LNG Terminal Sendout ────────────────────────────────────────────────────

def lng_status(wow_change_pct: float | None, today: date | None = None) -> tuple[str, str, str]:
    if wow_change_pct is None:
        return ("ok", "LNG sendout: awaiting data", "")
    injecting = _injection_season(today or date.today())
    if wow_change_pct < -15:
        return (
            "critical",
            f"LNG sendout down {abs(wow_change_pct):.1f}% week-on-week, suggesting potential supply disruption",
            (
                "A sendout drop of this magnitude may indicate a terminal outage, cargo diversion to Asia, "
                "or reduced US export volumes. Monitor Zeebrugge, Gate, and South Hook specifically. "
                "If sustained, this would tighten the European gas balance and support TTF prices."
            ),
        )
    elif wow_change_pct < -7:
        return (
            "warn",
            f"LNG sendout down {abs(wow_change_pct):.1f}% week-on-week",
            (
                "Moderate decline in NW European LNG sendout. Watch for continuation; "
                + (
                    "a sustained drop would slow storage injection and tighten the gas balance."
                    if injecting else
                    "a sustained drop would deepen storage withdrawals and tighten the winter gas balance."
                )
            ),
        )
    elif wow_change_pct > 10:
        return (
            "good",
            f"LNG sendout up {wow_change_pct:.1f}% week-on-week, "
            + ("supportive for the refill season" if injecting else "supportive for winter supply"),
            (
                ("Strong LNG imports are supporting storage injection. " if injecting else
                 "Strong LNG imports are cushioning storage withdrawals. ")
                + "Cargo arrivals into North-West European terminals are running well above the prior week."
            ),
        )
    else:
        return (
            "ok",
            f"LNG sendout stable ({wow_change_pct:+.1f}% versus prior 7 days)",
            "NW European LNG sendout is broadly stable. No acute supply signal.",
        )


# ── Market Summary ───────────────────────────────────────────────────────────

def market_summary(
    eu_pct: float, eu_min: float | None, eu_mean: float | None,
    de_pct: float | None, de_min: float | None, de_mean: float | None,
    ttf_price: float, ttf_ma30: float,
    nordic_avg: float | None, continental_avg: float | None,
) -> str:
    """Generate a one-paragraph market commentary from live data."""
    parts = []

    # Storage
    if eu_mean is not None and eu_min is not None:
        if eu_pct < eu_min:
            parts.append(
                f"European gas storage stands at a 5-year low of <strong>{eu_pct:.1f}%</strong>, "
                f"below the prior minimum of {eu_min:.1f}% for this date."
            )
        else:
            gap = eu_pct - eu_mean
            parts.append(
                f"European gas storage is at <strong>{eu_pct:.1f}%</strong>, "
                f"{abs(gap):.1f}pp {'above' if gap >= 0 else 'below'} the 5-year average."
            )
    else:
        parts.append(f"European gas storage is at <strong>{eu_pct:.1f}%</strong>.")

    if de_pct is not None and de_min is not None and de_pct < de_min:
        parts.append(
            f"Germany is particularly exposed at <strong>{de_pct:.1f}%</strong>, already below the 5-year minimum "
            f"({de_min:.1f}%), the weakest level for this date in the last five years."
        )

    # TTF
    ttf_vs_ma = (ttf_price / ttf_ma30 - 1) * 100
    if ttf_price > 45:
        parts.append(
            f"TTF is trading at <strong>€{ttf_price:.2f}/MWh</strong>, elevated relative to the "
            f"pre-crisis long-run average, reflecting storage tightness and refill demand."
        )
    else:
        parts.append(
            f"TTF is at <strong>€{ttf_price:.2f}/MWh</strong> ({ttf_vs_ma:+.1f}% versus the 30-day average)."
        )

    # Spread
    if nordic_avg is not None and continental_avg is not None:
        spread = continental_avg - nordic_avg
        if abs(spread) > 8:
            direction = "above" if spread > 0 else "below"
            parts.append(
                f"NL day-ahead prices are <strong>€{abs(spread):.0f}/MWh {direction}</strong> the Nordic average "
                f"(DE-LU unavailable via Nord Pool; NL used as continental proxy), "
                f"{'incentivising Norwegian hydro exports' if spread > 0 else 'an unusual reversal suggesting Nordic supply stress'}."
            )

    return " ".join(parts)
