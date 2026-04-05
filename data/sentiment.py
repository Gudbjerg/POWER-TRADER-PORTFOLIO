"""
FinBERT energy news sentiment pipeline.

Fetches energy-relevant RSS headlines, classifies each headline as
positive/negative/neutral using ProsusAI/finbert, and aggregates to a
daily net sentiment score.

Soft dependencies: feedparser, transformers, torch.
All are caught with graceful fallback messages if not installed.
"""
from __future__ import annotations

import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone

ENERGY_KEYWORDS = [
    "lng", "ttf", "natural gas", "gas price", "gas storage",
    "pipeline", "gazprom", "nord stream", "energy crisis",
    "power price", "electricity price", "wind power", "solar power",
    "carbon price", "ets", "emissions trading", "oil price",
    "crude oil", "brent", "hormuz", "opec", "nuclear power",
    "coal price", "equinor", "statoil", "norway gas", "norwegian gas",
    "european gas", "energy supply", "energy market", "strait of hormuz",
    "gas supply", "power market", "renewables", "offshore",
]

# Fetched via requests (not feedparser URL) to avoid SSL cert issues on macOS Python.
RSS_FEEDS = [
    ("BBC Business",    "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("Guardian Energy", "https://www.theguardian.com/environment/energy/rss"),
    ("LNG World News",  "https://www.lngworldnews.com/feed/"),
    ("Energy Monitor",  "https://www.energymonitor.ai/feed/"),
]


@st.cache_resource(show_spinner="Loading FinBERT model (first run only)…")
def _load_finbert():
    """
    Load ProsusAI/finbert pipeline once. Cached for the lifetime of the process.
    Returns (pipeline, None) on success or (None, error_message) on failure.
    """
    try:
        import logging
        import os
        # Suppress cosmetic warnings:
        # - "bert.embeddings.position_ids UNEXPECTED" is a known FinBERT checkpoint artifact, safe to ignore
        # - Unauthenticated HF Hub requests warning is about rate limits only, not an error
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from transformers import pipeline as hf_pipeline
        clf = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,   # return all three class scores
            device=-1,    # CPU
        )
        return clf, None
    except ImportError:
        return None, "transformers / torch not installed. Run: pip3 install transformers torch"
    except Exception as e:
        return None, f"FinBERT load failed: {e}"


def _is_energy_relevant(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in ENERGY_KEYWORDS)


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_energy_headlines(max_per_feed: int = 60) -> list[dict]:
    """
    Fetch energy-relevant headlines from public RSS feeds.
    Uses requests to fetch content (avoids macOS SSL cert issues), then
    passes bytes to feedparser for parsing.
    Returns list of {title, date, source} dicts, sorted newest first.
    """
    try:
        import feedparser
        import requests as _req
    except ImportError:
        return []

    articles: list[dict] = []
    for source, url in RSS_FEEDS:
        try:
            resp = _req.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:max_per_feed]:
                title   = entry.get("title", "").strip()
                summary = entry.get("summary", "")
                if not title:
                    continue
                if not _is_energy_relevant(title + " " + summary):
                    continue
                pub = entry.get("published_parsed")
                if pub:
                    dt = datetime(*pub[:6], tzinfo=timezone.utc).replace(tzinfo=None)
                else:
                    dt = datetime.utcnow()
                articles.append({"title": title, "date": dt.date(), "source": source})
        except Exception:
            continue

    # Deduplicate by title
    seen: set = set()
    unique = []
    for a in articles:
        if a["title"] not in seen:
            seen.add(a["title"])
            unique.append(a)

    # Drop stale articles (RSS feeds sometimes return cached 2024 entries)
    cutoff = (datetime.utcnow() - timedelta(days=30)).date()
    unique = [a for a in unique if a["date"] >= cutoff]

    return sorted(unique, key=lambda x: x["date"], reverse=True)


@st.cache_data(ttl=21600, show_spinner=False)
def score_headlines(headlines: list[dict]) -> pd.DataFrame:
    """
    Classify each headline with FinBERT.
    Returns DataFrame: date, title, source, pos, neg, neu, net_sentiment.
    """
    clf, err = _load_finbert()
    if clf is None:
        return pd.DataFrame()

    records = []
    for h in headlines:
        try:
            result = clf(h["title"][:512])
            # result is [[{label: score}, ...]]
            scores = {r["label"]: r["score"] for r in result[0]}
            records.append({
                "date":          pd.Timestamp(h["date"]),
                "title":         h["title"],
                "source":        h["source"],
                "pos":           scores.get("positive", 0.0),
                "neg":           scores.get("negative", 0.0),
                "neu":           scores.get("neutral",  0.0),
                "net_sentiment": scores.get("positive", 0.0) - scores.get("negative", 0.0),
            })
        except Exception:
            continue

    return pd.DataFrame(records) if records else pd.DataFrame()


_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", ".sentiment_history.csv")
_HISTORY_DAYS = 60  # keep rolling 60-day window on disk


def _load_history() -> pd.DataFrame:
    """Load persisted scored headlines from disk (survives app restarts)."""
    try:
        path = os.path.normpath(_HISTORY_FILE)
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, parse_dates=["date"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=_HISTORY_DAYS)
        return df[df["date"] >= cutoff].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _save_history(scored: pd.DataFrame) -> None:
    """Merge new scored headlines into the persistent history file."""
    try:
        path    = os.path.normpath(_HISTORY_FILE)
        hist    = _load_history()
        combined = pd.concat([hist, scored], ignore_index=True)
        # Deduplicate by title (same article may appear across fetches)
        combined = combined.drop_duplicates(subset=["title"]).sort_values("date")
        # Keep only last _HISTORY_DAYS days
        cutoff  = pd.Timestamp.now() - pd.Timedelta(days=_HISTORY_DAYS)
        combined = combined[combined["date"] >= cutoff]
        combined.to_csv(path, index=False)
    except Exception:
        pass


def get_sentiment_data() -> dict:
    """
    Full pipeline: fetch → score → merge with persisted history → aggregate to daily.

    History is persisted to .sentiment_history.csv so the daily aggregation grows
    over time (RSS feeds only carry ~2-7 days of articles per fetch).

    Returns dict with keys:
        headlines   : list[dict]        — raw filtered headlines (current fetch only)
        scored      : pd.DataFrame      — per-headline scores (full history window)
        daily       : pd.DataFrame      — daily aggregated (date, net_sentiment, ma3, ma7, n)
        error       : str | None        — human-readable error if pipeline failed
        fetched_at  : datetime
    """
    headlines = fetch_energy_headlines()

    if not headlines:
        # Even with no new articles, return history if we have it
        hist = _load_history()
        if hist.empty:
            return {
                "headlines": [],
                "scored":    pd.DataFrame(),
                "daily":     pd.DataFrame(),
                "error":     "No energy headlines fetched. RSS feeds may be unreachable.",
                "fetched_at": datetime.utcnow(),
            }
        scored = hist
    else:
        fresh = score_headlines(headlines)
        if not fresh.empty:
            _save_history(fresh)
        # Merge fresh scores with persisted history for the full daily series
        hist   = _load_history()
        scored = pd.concat([hist, fresh], ignore_index=True).drop_duplicates(subset=["title"])

    if scored.empty:
        _, err = _load_finbert()
        return {
            "headlines": headlines,
            "scored":    pd.DataFrame(),
            "daily":     pd.DataFrame(),
            "error":     err or "Scoring produced no results.",
            "fetched_at": datetime.utcnow(),
        }

    scored["date"] = pd.to_datetime(scored["date"])

    daily = (
        scored
        .groupby("date")
        .agg(
            net_sentiment=("net_sentiment", "mean"),
            n_headlines=("net_sentiment", "count"),
            avg_pos=("pos", "mean"),
            avg_neg=("neg", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )
    daily["ma3"] = daily["net_sentiment"].rolling(3, min_periods=1).mean()
    daily["ma7"] = daily["net_sentiment"].rolling(7, min_periods=1).mean()

    return {
        "headlines":  headlines,
        "scored":     scored,
        "daily":      daily,
        "error":      None,
        "fetched_at": datetime.utcnow(),
    }
