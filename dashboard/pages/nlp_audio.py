from __future__ import annotations
import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.loader import get_audio_features, get_all_audio_nlp
from data.transforms import safe_float
from components.artist_header import artist_header
from data.loader import get_artist, get_albums, get_tracks, get_artist_url

from components.charts import correlation_heatmap, scatter_ttr_streams
from components.filters import artist_selector
from config import (AUDIO_FEATURES, AUDIO_FEATURES_DISPLAY,
                    NLP_FEATURES, NLP_FEATURES_DISPLAY, COLORS)


_NLP_SHORT = {
    "ttr":              "TTR",
    "rhyme_density":    "Rimes",
    "semantic_density": "Sémantique",
    "sentiment_negative": "Sent. −",
    "avg_word_length":  "Long. mot",
    "lexical_diversity":"Diversité lex.",
    "hapax_ratio":      "Hapax",
}


def render():
    st.title("🔊 NLP × Audio")
    
    artist_name = artist_selector(key="nlp_audio_artist")
    artist = get_artist(artist_name)
    artist_image_url = get_artist_url(artist_name)
    artist_header(artist, artist_image_url)

    mode = st.radio("Périmètre", ["Un artiste", "Corpus complet"],
                    horizontal=True, key="nlp_audio_mode")

    if mode == "Un artiste":
        if not artist_name:
            st.info("Sélectionne un artiste.")
            return
        df = get_audio_features(artist_name)
        if df.empty:
            st.warning("Aucune feature audio pour cet artiste (table audio_features_local vide ou non jointe).")
            return
    else:
        df = get_all_audio_nlp()
        if df.empty:
            st.warning("Table audio_features_local vide ou non disponible.")
            return

    st.caption(f"{len(df)} pistes avec features audio + NLP")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Heatmap de corrélation ─────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Matrice de corrélation NLP × Audio</div>',
                unsafe_allow_html=True)
    st.caption("Vert = corrélation positive · Rouge = corrélation négative · Blanc = neutre")

    audio_avail = [c for c in AUDIO_FEATURES if c in df.columns]
    nlp_avail   = [c for c in _NLP_SHORT if c in df.columns]
    all_cols    = audio_avail + nlp_avail
    all_labels  = {**AUDIO_FEATURES_DISPLAY, **_NLP_SHORT}

    if len(all_cols) >= 4:
        st.plotly_chart(
            correlation_heatmap(df, all_cols, all_labels),
            width='stretch'
        )
    else:
        st.info("Pas assez de colonnes pour la corrélation.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Scatters spécifiques ───────────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="small")

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if "rhyme_density" in df.columns and "tempo" in df.columns:
            st.markdown('<div class="card-title">Densité de rimes vs Tempo</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                scatter_ttr_streams(df, "rhyme_density", "tempo",
                                     "Densité rimes", "Tempo (BPM)"),
                width='stretch'
            )
        else:
            st.info("Colonnes rhyme_density / tempo absentes.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if "sentiment_negative" in df.columns and "roughness" in df.columns:
            st.markdown('<div class="card-title">Sentiment négatif vs Rugosité audio</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                scatter_ttr_streams(df, "sentiment_negative", "roughness",
                                     "Sentiment négatif", "Rugosité"),
                width='stretch'
            )
        else:
            st.info("Colonnes sentiment_negative / roughness absentes.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Scatter personnalisé ───────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Scatter personnalisé</div>', unsafe_allow_html=True)

    all_numeric = [c for c in df.select_dtypes("number").columns
                   if c not in ("track_id", "artist_id", "album_id", "key", "mode")]
    if len(all_numeric) >= 2:
        c1, c2 = st.columns(2, gap="small")
        with c1:
            x_col = st.selectbox("Axe X", all_numeric, key="custom_x",
                                  format_func=lambda c: all_labels.get(c, c))
        with c2:
            default_y = all_numeric[1] if len(all_numeric) > 1 else all_numeric[0]
            y_col = st.selectbox("Axe Y", all_numeric,
                                  index=min(1, len(all_numeric)-1),
                                  key="custom_y",
                                  format_func=lambda c: all_labels.get(c, c))
        hover = "track_name" if "track_name" in df.columns else None
        import plotly.express as px
        fig = px.scatter(
            df.dropna(subset=[x_col, y_col]),
            x=x_col, y=y_col,
            hover_name=hover,
            color_discrete_sequence=[COLORS["primary"]],
            trendline="ols",
            trendline_color_override="#5dbf8a",
        )
        fig.update_traces(marker=dict(size=6, opacity=0.65))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=8, r=8, t=8, b=8),
            font=dict(family="DM Sans", size=11),
            xaxis=dict(title=all_labels.get(x_col, x_col), gridcolor="#f0f0f0"),
            yaxis=dict(title=all_labels.get(y_col, y_col), gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Pas assez de colonnes numériques.")
    st.markdown('</div>', unsafe_allow_html=True)
