"""Layout guards for the router and the page modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_legacy_pages_directory():
    """A pages/ folder next to app.py, even one holding only __pycache__, makes Streamlit
    run the first request after a server start in legacy multipage mode: a deep link such
    as /Energy_Globe then shows "Page not found" before the router takes over. Page modules
    live in views/."""
    assert not (ROOT / "pages").exists(), (
        "Remove the leftover pages/ folder next to app.py; page modules live in views/."
    )


def test_every_view_is_routed():
    router = (ROOT / "app.py").read_text(encoding="utf-8")
    views = sorted((ROOT / "views").glob("[0-9]_*.py"))
    assert views, "no page modules found in views/"
    for view in views:
        assert f'st.Page("views/{view.name}"' in router, f"{view.name} is not registered in app.py"
