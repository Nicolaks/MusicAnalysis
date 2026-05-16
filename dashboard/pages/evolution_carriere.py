from __future__ import annotations
import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.loader import get_albums
from data.transforms import safe_float
from components.charts import (sentiment_line, emotion_heatmap,
                                vocab_evolution, emotion_lines)
from components.filters import artist_selector
from config import LEXICAL_FIELD_DISPLAY
from config import EMOTION_LABELS
from data.loader import get_artist, get_albums, get_tracks, get_artist_url
from components.artist_header import artist_header
import plotly.graph_objects as go
from components.charts import _LAYOUT


def _lexical_area(df):
    cols_map = {
        "avg_lexical_violence": "Violence",
        "avg_lexical_street":   "Street",
        "avg_lexical_love":     "Amour",
        "avg_lexical_money":    "Argent",
    }
    avail = {v: k for k, v in cols_map.items() if k in df.columns}
    if not avail:
        return go.Figure()
    colors = ["#a32d2d", "#1a5c38", "#185fa5", "#854f0b"]
    fig = go.Figure()
    df2 = df.sort_values("release_year", na_position="last")
    for (label, col), color in zip(avail.items(), colors):
        fig.add_trace(go.Scatter(
            x=df2["album_name"], y=df2[col],
            name=label, mode="lines+markers",
            stackgroup="one",
            line=dict(color=color, width=1.5),
            fillcolor=color + "55",
        ))
    fig.update_layout(
        **_LAYOUT, height=260,
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#f0f0f0", tickformat=".3f"),
        legend=dict(orientation="h", y=-0.35, font=dict(size=10)),
    )
    return fig


def _flesch_line(df):
    if "avg_flesch_kincaid_grade" not in df.columns:
        return go.Figure()
    df2 = df.sort_values("release_year", na_position="last")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df2["album_name"], y=df2["avg_flesch_kincaid_grade"],
        marker=dict(
            color=df2["avg_flesch_kincaid_grade"],
            colorscale=[[0, "#e8f5ee"], [1, "#0f3d25"]],
            showscale=False,
        ),
        text=df2["avg_flesch_kincaid_grade"].round(1),
        textposition="outside",
        textfont=dict(size=9),
    ))
    fig.update_layout(
        **_LAYOUT, height=240,
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#f0f0f0", title="Grade"),
    )
    return fig


def render():
    st.title("📅 Évolution carrière")

    artist_name = artist_selector(key="evol_artist")
    
    artist = get_artist(artist_name)
    artist_image_url = get_artist_url(artist_name)
    artist_header(artist, artist_image_url)
    if not artist_name:
        st.info("Sélectionne un artiste.")
        return

    albums = get_albums(artist_name)
    if albums.empty:
        st.warning("Aucun album disponible.")
        return

    st.caption(f"{len(albums)} albums · {artist_name}")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Sentiment + Vocabulaire ────────────────────────────────────────────────
    col1 = st.container()

    with col1:
        st.markdown('<div class="card-title">Évolution des 4 émotions dominantes de l\'artiste par album</div>', unsafe_allow_html=True)
        has_emo = "avg_emotion_scores" in albums.columns and albums["avg_emotion_scores"].notna().any()
        if has_emo:
            st.plotly_chart(emotion_lines(albums), use_container_width=True)
        else:
            st.info("Données émotions absentes.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    col2 = st.container()

    with col2:
        st.markdown('<div class="card-title">Évolution de la richesse lexicale au fil des albums</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(vocab_evolution(albums), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    st.info("💡 Glissez la barre ci-dessus pour naviguer dans la discographie.")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Heatmap émotions ──────────────────────────────────────────────────────
    st.markdown('<div class="card-title">Carte émotionnelle — album par album</div>',
                unsafe_allow_html=True)
    emotion_cols = [c for c in EMOTION_LABELS if c in albums.columns]

    if emotion_cols:
        df_heat = albums.set_index("album_name")[emotion_cols]

        st.plotly_chart(
            emotion_heatmap(df_heat),
            width="stretch"
        )
    else:
        st.info("Colonnes émotions absentes.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Champs lexicaux + Lisibilité ──────────────────────────────────────────
    col3 = st.container()

    with col3:
        st.markdown('<div class="card-title">Évolution des champs lexicaux</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_lexical_area(albums), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    col4 = st.container()

    with col4:
        st.markdown('<div class="card-title">Complexité des textes (Flesch-Kincaid)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_flesch_line(albums), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tableau récapitulatif ──────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">Tableau récapitulatif albums</div>',
                unsafe_allow_html=True)
    display_cols = {
        "album_name":             "Album",
        "release_year":           "Année",
        "track_count":            "Titres",
        "album_vocabulary_size":  "Vocabulaire",
        "album_ttr":              "TTR",
        "avg_rhyme_density":      "Densité rimes",
        "avg_sentiment_positive": "Sentiment +",
        "avg_sentiment_negative": "Sentiment −",
        "avg_flesch_kincaid_grade": "FK Grade",
    }
    avail = {k: v for k, v in display_cols.items() if k in albums.columns}
    if avail:
        shown = albums[list(avail.keys())].rename(columns=avail)
        st.dataframe(shown, width="stretch", hide_index=True,
                     column_config={
                         "TTR":          st.column_config.NumberColumn(format="%.3f"),
                         "Densité rimes":st.column_config.NumberColumn(format="%.3f"),
                         "Sentiment +":  st.column_config.NumberColumn(format="%.3f"),
                         "Sentiment −":  st.column_config.NumberColumn(format="%.3f"),
                         "FK Grade":     st.column_config.NumberColumn(format="%.1f"),
                     })
    st.markdown('</div>', unsafe_allow_html=True)
