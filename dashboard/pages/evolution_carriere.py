from __future__ import annotations
import streamlit as st
import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.loader import get_albums
from data.transforms import safe_float, normalize_fk_grade
from components.charts import (sentiment_line, emotion_heatmap,
                                vocab_evolution, emotion_lines, emotion_stacked_bars, lexical_area)
from components.filters import artist_selector
from config import LEXICAL_FIELD_DISPLAY
from config import EMOTION_LABELS
from data.loader import get_artist, get_albums, get_tracks, get_artist_url
from components.artist_header import artist_header
import plotly.graph_objects as go
from components.charts import _LAYOUT



def _flesch_line(df):
    if "avg_flesch_kincaid_grade" not in df.columns:
        return go.Figure()
    df2 = df.sort_values("release_year", na_position="last").copy()
    df2 = df2[df2["avg_flesch_kincaid_grade"] != 0].dropna(subset=["avg_flesch_kincaid_grade"])
    
    if df2.empty:
        return go.Figure()
    raw = df2["avg_flesch_kincaid_grade"]
    df2["accessibilite"] = raw.apply(
        lambda x: normalize_fk_grade(x, raw.min(), raw.max())
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df2["album_name"],
        y=df2["accessibilite"],
        marker=dict(
            color=df2["accessibilite"],
            colorscale=[[0, "#e8f5ee"], [1, "#0f3d25"]],
            showscale=False,
        ),
        text=df2["accessibilite"].apply(lambda x: f"{x:.2f}"),
        textposition="outside",
        textfont=dict(size=9),
        customdata=raw.round(1),
        hovertemplate="<b>%{x}</b><br>Accessibilité : %{y:.2f}<br>FK brut : %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT, height=400,
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(
            gridcolor="#f0f0f0",
            title="Accessibilité (0 = complexe, 1 = accessible)",
            range=[0, 1.15],
        ),
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
            st.plotly_chart(emotion_lines(albums), width='stretch')
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
    st.markdown('<div class="card-title">Barre émotionnelle (8 émotions principales) : album par album</div>',
                unsafe_allow_html=True)
    has_emo = "avg_emotion_scores" in albums.columns and albums["avg_emotion_scores"].notna().any()
    if has_emo:
        st.plotly_chart(emotion_stacked_bars(albums), width='stretch')
    else:
        st.info("Colonnes émotions absentes.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Champs lexicaux + Lisibilité ──────────────────────────────────────────
    col3 = st.container()

    with col3:
        st.markdown('<div class="card-title">Évolution des champs lexicaux</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(lexical_area(albums), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    col4 = st.container()

    with col4:
        st.markdown('<div class="card-title">Accessibilité des textes</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_flesch_line(albums), width="stretch")
        with st.expander("Comment lire ce graphique ?"):
            st.markdown("""
            **0 = texte dense et complexe** : phrases longues, mots polysyllabiques, style très oral.  
            **1 = texte accessible** : phrases courtes, vocabulaire simple, facile à suivre.
            
            Plus la barre est haute, plus les paroles de l'album sont faciles à lire et à comprendre.  
            Un score faible ne signifie pas un texte "mauvais", il peut refléter un style technique, 
            un flow dense ou une écriture très compressée. 
            
            """)
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
        "avg_word_count":         "Moyenne de mots",
        "album_vocabulary_size":  "Nombre de mots dans l'album",
        "dominant_emotions":        "Emotions dominantes",
        "dominant_lexical_fields": "Champs léxicaux dominants",

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
