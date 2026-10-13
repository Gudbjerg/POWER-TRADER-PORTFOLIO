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
import threading
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
    ("BBC Business",        "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("Guardian Energy",     "https://www.theguardian.com/environment/energy/rss"),
    ("LNG World News",      "https://www.lngworldnews.com/feed/"),
    ("Energy Monitor",      "https://www.energymonitor.ai/feed/"),
    ("Power Technology",    "https://www.power-technology.com/feed/"),
    ("Offshore Technology", "https://www.offshore-technology.com/feed/"),
]

# Feeds that serve older items page by page (WordPress "?paged=N"). They let the history be
# rebuilt after a restart instead of waiting weeks for live headlines to accumulate.
PAGED_FEEDS = {"Energy Monitor", "Power Technology", "Offshore Technology"}
BACKFILL_MAX_PAGES = 30          # per feed
SENTIMENT_MIN_DAYS = 21          # days of history the Granger and IC tests need
BACKFILL_TARGET_DAYS = 60        # calendar days a backfill reaches back
FEED_SOURCES_TEXT = ", ".join(name for name, _ in RSS_FEEDS)


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
        return None, ("The FinBERT model is not available in this environment "
                      "(PyTorch and Transformers are not installed).")
    except Exception as e:
        return None, f"The FinBERT model could not be loaded ({type(e).__name__})."


def _is_energy_relevant(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in ENERGY_KEYWORDS)


def _parse_feed(content: bytes, source: str, max_items: int | None = None) -> list[dict]:
    """Energy-relevant headlines from one RSS document: {title, date, source}."""
    import feedparser
    feed = feedparser.parse(content)
    out = []
    for entry in feed.entries[:max_items]:
        title   = entry.get("title", "").strip()
        summary = entry.get("summary", "")
        if not title or not _is_energy_relevant(title + " " + summary):
            continue
        pub = entry.get("published_parsed") or entry.get("updated_parsed")
        if not pub:
            continue
        dt = datetime(*pub[:6], tzinfo=timezone.utc).replace(tzinfo=None)
        out.append({"title": title, "date": dt.date(), "source": source})
    return out


def backfill_headlines(since: str) -> list[dict]:
    """Older energy headlines from the paged feeds, back to ``since`` (YYYY-MM-DD)."""
    try:
        import feedparser  # noqa: F401
        import requests as _req
    except ImportError:
        return []
    cutoff = datetime.strptime(since, "%Y-%m-%d").date()
    articles: list[dict] = []
    for source, url in RSS_FEEDS:
        if source not in PAGED_FEEDS:
            continue
        for page in range(1, BACKFILL_MAX_PAGES + 1):
            try:
                resp = _req.get(url, params={"paged": page}, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code != 200:
                    break
                import feedparser
                entries = feedparser.parse(resp.content).entries
                if not entries:
                    break
                articles.extend(a for a in _parse_feed(resp.content, source) if a["date"] >= cutoff)
                oldest = min((e.get("published_parsed") or e.get("updated_parsed")
                              for e in entries if e.get("published_parsed") or e.get("updated_parsed")),
                             default=None)
                if oldest is None or datetime(*oldest[:3]).date() < cutoff:
                    break
            except Exception:
                break
    seen: set = set()
    unique = []
    for a in articles:
        if a["title"] not in seen:
            seen.add(a["title"])
            unique.append(a)
    return sorted(unique, key=lambda x: x["date"], reverse=True)


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
            articles.extend(_parse_feed(resp.content, source, max_per_feed))
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
    return _score_with(clf, headlines)


def _score_with(clf, headlines: list[dict]) -> pd.DataFrame:
    """Score headlines with a loaded FinBERT pipeline (no Streamlit calls)."""
    records = []
    titles = [h["title"][:512] for h in headlines]
    try:
        results = clf(titles, batch_size=16, truncation=True) if titles else []
    except Exception:
        results = []
    for h, result in zip(headlines, results):
        try:
            # result is [{label, score}, ...] for each of the three classes
            scores = {r["label"]: r["score"] for r in result}
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


_PERSIST_DIR   = os.getenv("DATA_PERSIST_DIR", os.path.join(os.path.dirname(__file__), ".."))
_HISTORY_FILE  = os.path.join(_PERSIST_DIR, ".sentiment_history.csv")
_HISTORY_DAYS = 120  # rolling window kept on disk (calendar days)
_HISTORY_LOCK = threading.Lock()


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
    with _HISTORY_LOCK:
        _save_history_locked(scored)


def _save_history_locked(scored: pd.DataFrame) -> None:
    try:
        path    = os.path.normpath(_HISTORY_FILE)
        if os.getenv("DATA_PERSIST_DIR"):
            import sys
            print(f"[sentiment] writing history to {path}", file=sys.stderr)
        hist    = _load_history()
        combined = pd.concat([hist, scored], ignore_index=True)
        combined = combined.drop_duplicates(subset=["title"]).sort_values("date")
        cutoff  = pd.Timestamp.now() - pd.Timedelta(days=_HISTORY_DAYS)
        combined = combined[combined["date"] >= cutoff]
        combined.to_csv(path, index=False)
    except Exception:
        pass


def _history_days(hist: pd.DataFrame, window_days: int = BACKFILL_TARGET_DAYS) -> int:
    if hist.empty or "date" not in hist.columns:
        return 0
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=window_days)
    return int(pd.to_datetime(hist["date"]).loc[lambda d: d >= cutoff].dt.normalize().nunique())


_BACKFILL_LOCK = threading.Lock()
_BACKFILL_STATE = {"running": False, "done": False}


def _run_backfill(clf) -> None:
    try:
        since = (datetime.utcnow() - timedelta(days=BACKFILL_TARGET_DAYS)).strftime("%Y-%m-%d")
        older = backfill_headlines(since)
        if older:
            scored = _score_with(clf, older)
            if not scored.empty:
                _save_history(scored)
    finally:
        with _BACKFILL_LOCK:
            _BACKFILL_STATE.update(running=False, done=True)


def backfill_running() -> bool:
    return _BACKFILL_STATE["running"]


def _ensure_history() -> None:
    """Start a background backfill when the recent history is too short for the tests.

    Paging the archives takes a minute or two, so it runs off the page's thread once per
    process; the page shows the history it has and picks up the backfill on a later run.
    """
    if _history_days(_load_history()) >= SENTIMENT_MIN_DAYS:
        return
    with _BACKFILL_LOCK:
        if _BACKFILL_STATE["running"] or _BACKFILL_STATE["done"]:
            return
        clf, _ = _load_finbert()
        if clf is None:
            return
        _BACKFILL_STATE["running"] = True
    threading.Thread(target=_run_backfill, args=(clf,), daemon=True).start()


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
    _ensure_history()

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
