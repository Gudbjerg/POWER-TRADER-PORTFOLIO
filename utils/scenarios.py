"""
Scenario interpretation logic. Translates raw data into trader-readable signals.
Returns (status, headline, detail) tuples where status is one of: critical, warn, ok, good.
"""
from __future__ import annotations
import pandas as pd


# ── Gas Storage ─────────────────────────────────────────────────────────────

def storage_status(pct: float, min_pct: float | None, mean_pct: float | None, region: str = "EU") -> tuple[str, str, str]:
    if mean_pct is None or min_pct is None:
        return ("ok", f"{region} storage at {pct:.1f}%; historical bands unavailable", "")
    gap_to_mean = pct - mean_pct
    gap_to_min  = pct - min_pct

    if pct < min_pct:
        return (
            "critical",
            f"{region} storage at historic low: {pct:.1f}% (below 5-year minimum of {min_pct:.1f}%)",
            (
                f"Storage has fallen below every prior year's level for this date over the last five years. "
                f"At {pct:.1f}%, the market faces a steep injection requirement heading into summer. "
                f"A cold snap or LNG supply disruption could rapidly tighten balances. "
                f"Watch EU injection mandate (90% by November 1). The gap to target is significant."
            ),
        )
    elif gap_to_mean < -15:
        return (
            "warn",
            f"{region} storage critically below the seasonal average: {pct:.1f}% versus {mean_pct:.1f}%",
            (
                f"Storage is {abs(gap_to_mean):.1f}pp below the 5-year average for this date. "
                f"While above the historical minimum, the pace of injection required to reach "
                f"80% by November 1 is well above seasonal norms. "
                f"Higher summer TTF prices are likely to incentivise LNG cargoes and pipeline volumes."
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

def lng_status(wow_change_pct: float | None) -> tuple[str, str, str]:
    if wow_change_pct is None:
        return ("ok", "LNG sendout: awaiting data", "")
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
                "a sustained drop would tighten the gas balance heading into injection season."
            ),
        )
    elif wow_change_pct > 10:
        return (
            "good",
            f"LNG sendout up {wow_change_pct:.1f}% week-on-week, supportive for the refill season",
            (
                "Strong LNG imports are supporting the injection season. "
                "US and Qatari cargoes appear to be flowing freely into NW Europe."
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
            f"({de_min:.1f}%), the weakest level for this date on record."
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
