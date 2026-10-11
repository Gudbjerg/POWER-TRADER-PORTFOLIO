"""
Shared pytest configuration.

- Puts the repository root on sys.path so tests import project modules
  exactly as the Streamlit pages do (``from data...``, ``from models...``).
- Registers the ``network`` marker for smoke tests that fetch live data.
  Skip them offline with:  pytest -m "not network"
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: test performs live HTTP requests (deselect with -m 'not network')",
    )
