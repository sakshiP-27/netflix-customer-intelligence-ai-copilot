"""Netflix-inspired look for every page. Visual only; it does not change any number."""

from __future__ import annotations

import html

import streamlit as st

_CSS = """
<style>
  .stApp {
    background:
      radial-gradient(900px 420px at 88% -10%, rgba(229, 9, 20, 0.45), transparent 60%),
      radial-gradient(700px 380px at -10% 20%, rgba(168, 85, 247, 0.18), transparent 55%),
      #141414;
    color: #ffffff;
  }
  [data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0.55);
    border-bottom: 3px solid #E50914;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a1a 0%, #050505 100%);
    border-right: 1px solid #2a2a2a;
  }
  [data-testid="stSidebar"]::before {
    content: "CUSTOMER INTELLIGENCE";
    display: block;
    color: #E50914;
    font-family: Impact, "Arial Black", sans-serif;
    font-size: 1.35rem;
    letter-spacing: 0.08em;
    line-height: 1.05;
    padding: 1.1rem 1.15rem 0.2rem;
  }
  [data-testid="stSidebarNav"] a {
    border-radius: 6px;
    margin: 2px 8px;
  }
  [data-testid="stSidebarNav"] a[aria-current="page"] {
    background: #E50914 !important;
    color: #ffffff !important;
  }
  .block-container {
    padding-top: 4.6rem;
    max-width: 1180px;
  }
  .nfx-hero {
    position: relative;
    overflow: hidden;
    border-radius: 14px;
    padding: 2.2rem 2rem 1.6rem;
    margin-bottom: 1.1rem;
    background:
      linear-gradient(100deg, rgba(0, 0, 0, 0.2) 0%, rgba(0, 0, 0, 0.72) 58%, rgba(0, 0, 0, 0.35) 100%),
      radial-gradient(420px 220px at 92% 20%, rgba(229, 9, 20, 0.95), transparent 70%),
      linear-gradient(160deg, #3a0008 0%, #141414 55%, #000 100%);
    border: 1px solid #3a0a10;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.45);
  }
  .nfx-hero::after {
    content: "";
    position: absolute;
    right: -40px;
    top: -60px;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255, 70, 80, 0.55), transparent 68%);
    pointer-events: none;
  }
  .nfx-kicker {
    position: relative;
    z-index: 1;
    color: #E50914;
    font-family: Impact, "Arial Black", sans-serif;
    letter-spacing: 0.22em;
    font-size: 0.82rem;
    margin-bottom: 0.35rem;
  }
  .nfx-hero h1 {
    position: relative;
    z-index: 1;
    margin: 0;
    color: #ffffff;
    font-family: Impact, "Arial Black", sans-serif;
    font-size: 3.4rem;
    letter-spacing: 0.03em;
    line-height: 0.95;
    text-shadow: 0 8px 24px rgba(0, 0, 0, 0.55);
  }
  .nfx-hero p {
    position: relative;
    z-index: 1;
    max-width: 720px;
    margin: 0.8rem 0 0;
    color: #e6e6e6;
    font-size: 1.02rem;
    line-height: 1.45;
  }
  .nfx-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.7rem;
    margin: 0 0 1.1rem;
  }
  .nfx-poster {
    min-height: 92px;
    border-radius: 10px;
    display: flex;
    align-items: flex-end;
    padding: 0.7rem;
    color: #ffffff;
    font-family: Impact, "Arial Black", sans-serif;
    letter-spacing: 0.06em;
    font-size: 1.15rem;
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.08);
  }
  .nfx-accent-0 { background: linear-gradient(160deg, #f5c518 0%, #7a4a00 100%); color: #1a1a1a; }
  .nfx-accent-1 { background: linear-gradient(160deg, #ff3b47 0%, #6d0010 100%); }
  .nfx-accent-2 { background: linear-gradient(160deg, #2ec4b6 0%, #0b3d45 100%); }
  .nfx-accent-3 { background: linear-gradient(160deg, #46d369 0%, #0d4a28 100%); }
  .nfx-accent-4 { background: linear-gradient(160deg, #b388ff 0%, #3b1578 100%); }
  .nfx-tile {
    background: linear-gradient(180deg, #242424 0%, #121212 100%);
    border: 1px solid #2e2e2e;
    border-top: 5px solid var(--accent, #E50914);
    border-radius: 12px;
    padding: 0.95rem 1rem 0.85rem;
    min-height: 138px;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28);
    margin-bottom: 0.4rem;
  }
  .nfx-tile-label {
    color: #b3b3b3;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .nfx-tile-value {
    color: #ffffff;
    font-family: Arial, Helvetica, sans-serif;
    font-weight: 800;
    font-size: 1.72rem;
    letter-spacing: 0;
    line-height: 1.15;
    margin: 0.3rem 0;
    white-space: nowrap;
  }
  .nfx-tile-note {
    color: #a3a3a3;
    font-size: 0.82rem;
    line-height: 1.35;
  }
  [data-testid="stMetric"] {
    background: linear-gradient(180deg, #242424, #121212);
    border: 1px solid #2e2e2e;
    border-top: 5px solid #E50914;
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
  }
  [data-testid="stMetricValue"] { color: #ffffff; }
  [data-testid="stPlotlyChart"] {
    background: linear-gradient(180deg, #1b1b1b, #101010);
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 0.35rem 0.35rem 0;
  }
  [data-testid="stForm"] {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 0.6rem 0.8rem 0.2rem;
  }
  .stButton > button, .stDownloadButton > button {
    background: #E50914;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    font-weight: 700;
  }
  .stButton > button:hover, .stDownloadButton > button:hover {
    background: #f40612;
    color: #ffffff;
    border: none;
  }
  div[data-baseweb="input"] > div,
  div[data-baseweb="select"] > div,
  div[data-baseweb="textarea"] > div {
    background: #1a1a1a;
    border-color: #3a3a3a;
  }
  [data-testid="stCaptionContainer"] { color: #b3b3b3; }
  h2, h3 { color: #ffffff; }
  footer { visibility: hidden; }
  @media (max-width: 900px) {
    .nfx-row { grid-template-columns: repeat(2, 1fr); }
    .nfx-hero h1 { font-size: 2.3rem; }
  }
</style>
"""

_POSTERS = (
    ("Customers", "nfx-accent-0"),
    ("Revenue", "nfx-accent-1"),
    ("Watching", "nfx-accent-2"),
    ("Churn", "nfx-accent-3"),
    ("Support", "nfx-accent-4"),
)

_TILE_ACCENTS = ("#F5C518", "#E50914", "#2EC4B6", "#46D369", "#B388FF", "#FF6B35")


def apply_theme() -> None:
    """Paint the black, red, and gradient frame used on every page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_heading(title: str, subtitle: str) -> None:
    """Poster-style title. The words are the page's own title and audience line."""
    st.markdown(
        (
            '<div class="nfx-hero">'
            '<div class="nfx-kicker">Now showing</div>'
            f"<h1>{html.escape(title)}</h1>"
            f"<p>{html.escape(subtitle)}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def poster_row() -> None:
    """Five gradient cards, in the same spirit as the colorful title row in the reference."""
    cards = "".join(
        f'<div class="nfx-poster {css}">{html.escape(label)}</div>' for label, css in _POSTERS
    )
    st.markdown(f'<div class="nfx-row">{cards}</div>', unsafe_allow_html=True)


def metric_tiles(tiles: list[tuple[str, str, str]]) -> None:
    """Show calculated figures as colored cards. Values are already formatted."""
    for start in range(0, len(tiles), 3):
        columns = st.columns(3)
        chunk = tiles[start : start + 3]
        for index, column in enumerate(columns):
            if index >= len(chunk):
                continue
            label, value, note = chunk[index]
            accent = _TILE_ACCENTS[(start + index) % len(_TILE_ACCENTS)]
            column.markdown(
                (
                    f'<div class="nfx-tile" style="--accent:{accent}">'
                    f'<div class="nfx-tile-label">{html.escape(label)}</div>'
                    f'<div class="nfx-tile-value">{html.escape(value)}</div>'
                    f'<div class="nfx-tile-note">{html.escape(note)}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
