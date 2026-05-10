from __future__ import annotations
import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.loader import get_artists


def artist_selector(key: str = "artist", label: str = "Artiste") -> str | None:
    artists = get_artists()
    if not artists:
        st.sidebar.warning("Aucun artiste en base.")
        return None
    return st.sidebar.selectbox(label, artists, key=key)


def multi_artist_selector(key: str = "artists", label: str = "Artistes", default_n: int = 3) -> list[str]:
    artists = get_artists()
    if not artists:
        return []
    default = artists[:min(default_n, len(artists))]
    return st.sidebar.multiselect(label, artists, default=default, key=key)


def year_range_slider(df: pd.DataFrame, col: str = "release_year", key: str = "years"):
    if df.empty or col not in df.columns or df[col].isna().all():
        return None, None
    mn, mx = int(df[col].min()), int(df[col].max())
    if mn == mx:
        return mn, mx
    return st.sidebar.slider("Période", mn, mx, (mn, mx), key=key)


def metric_selector(options: list[str], labels: dict[str, str] | None = None,
                    label: str = "Métrique", key: str = "metric") -> str:
    display = [labels[o] if labels and o in labels else o for o in options]
    idx = st.sidebar.selectbox(label, range(len(options)),
                                format_func=lambda i: display[i], key=key)
    return options[idx]
