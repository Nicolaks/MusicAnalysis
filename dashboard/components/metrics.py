from __future__ import annotations
import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.transforms import safe_float, streams_label


def kpi_row(items: list[dict]):
    """
    items = [{"label": str, "value": str, "sub": str, "featured": bool}]
    """
    cols = st.columns(len(items), gap="small")
    for col, item in zip(cols, items):
        featured = item.get("featured", False)
        cls = "kpi-featured" if featured else "kpi-card"
        col.markdown(f"""
<div class="{cls}">
  <div class="kpi-lbl">{item.get('label','')}</div>
  <div class="kpi-val">{item.get('value','—')}</div>
  <div class="kpi-sub">{item.get('sub','')}</div>
</div>""", unsafe_allow_html=True)


def artist_kpis(artist: pd.Series, stream: int):
    vocab  = int(safe_float(artist.get("career_vocabulary_size", 0)))
    ttr    = safe_float(artist.get("career_ttr", 0))
    albums = int(safe_float(artist.get("album_count", 0)))
    tracks = int(safe_float(artist.get("track_count", 0)))
    streams = streams_label(stream)

    kpi_row([
        {"label": "Vocabulaire carrière", "value": f"{vocab:,}",
         "sub": f"TTR global : {ttr:.3f}", "featured": True},
        {"label": "Albums analysés",  "value": str(albums), "sub": "discographie"},
        {"label": "Titres analysés",  "value": str(tracks), "sub": "paroles"},
        {"label": "Nombre de streams Spotify", "value": str(streams), "sub": "streams"},
    ])
