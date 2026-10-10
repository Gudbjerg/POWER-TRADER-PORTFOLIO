"""
European Gas & Power Market Intelligence Platform: application entry point.

``app.py`` is the router. It configures the page defaults, applies the design system
once per run, declares the grouped navigation and runs the selected page. Page modules
live in ``pages/``; their URL paths are fixed below so existing links (for example
``/Live_Monitor``) keep working.

Run with:  streamlit run app.py
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Shared defaults only. Each page sets its own title and sidebar state; setting them
# here as well would make the sidebar collapse on every rerun of pages that expand it.
st.set_page_config(page_icon="⚡", layout="wide")

from utils.theme import begin_run, inject_theme, render_logo  # noqa: E402

begin_run()
inject_theme()
render_logo()

# Section order and page order define the top navigation (and the sidebar menu on
# narrow screens, where Streamlit moves top navigation into the sidebar).
PAGES = {
    # Overview: the landing page sits directly in the bar rather than in a one-item menu.
    "": [
        st.Page("pages/0_Home.py", title="Overview", icon=":material/space_dashboard:", default=True),
    ],
    "Markets": [
        st.Page("pages/1_Live_Monitor.py", title="Live Monitor",
                icon=":material/monitoring:", url_path="Live_Monitor"),
        st.Page("pages/3_Macro_Signals.py", title="Macro Signals",
                icon=":material/public:", url_path="Macro_Signals"),
    ],
    "Analytics": [
        st.Page("pages/2_Quant_Analysis.py", title="Quant Analysis",
                icon=":material/query_stats:", url_path="Quant_Analysis"),
        st.Page("pages/4_ML_Models.py", title="ML Models",
                icon=":material/memory:", url_path="ML_Models"),
    ],
    "Signals & Risk": [
        st.Page("pages/5_Mispricing_Dashboard.py", title="Mispricing Dashboard",
                icon=":material/track_changes:", url_path="Mispricing_Dashboard"),
        st.Page("pages/6_Risk_Dashboard.py", title="Risk Dashboard",
                icon=":material/shield:", url_path="Risk_Dashboard"),
    ],
    "Projects": [
        st.Page("pages/7_BESS_Optimiser.py", title="BESS Optimiser",
                icon=":material/battery_charging_full:", url_path="BESS_Optimiser"),
    ],
}

pg = st.navigation(PAGES, position="top")
pg.run()
