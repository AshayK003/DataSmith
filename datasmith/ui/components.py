"""Shared UI components for DataSmith."""
import streamlit as st

from datasmith.ui import icons


def render_header(active: str = "home") -> None:
    """Render the top navigation bar with brand and page links.

    Call this at the top of every page, BEFORE any other content.
    """
    pages = [
        ("home", "Home", "/"),
        ("generate", "Generate", "/Generate"),
        ("about", "About", "/About"),
    ]
    nav_html = ""
    for key, label, href in pages:
        cls = "active" if key == active else ""
        nav_html += f'<a class="ds-nav-btn {cls}" href="{href}" title="{label}">{label}</a>'

    st.markdown(
        '<div class="ds-header">'
        f'<span class="ds-header-brand">{icons.BRAND} DataSmith</span>'
        f'<div class="ds-header-nav">{nav_html}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
