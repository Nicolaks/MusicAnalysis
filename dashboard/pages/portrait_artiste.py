from __future__ import annotations
import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.loader import get_artist, get_albums, get_tracks, get_artist_url
from data.transforms import (parse_top_words, parse_style_signature,
                              normalize_radar, albums_emotion_matrix,
                              safe_float, streams_label)
from components.charts import (radar_chart, top_words_bar, emotion_heatmap,
                                sentiment_donut, lexical_bars)
from components.metrics import artist_kpis
from components.filters import artist_selector
from config import RADAR_KEYS, RADAR_DISPLAY, LEXICAL_FIELD_DISPLAY, EMOTION_DISPLAY
from components.artist_header import artist_header


def render():
    st.title("🎤 Portrait artiste")

    artist_name = artist_selector(key="portrait_artist")
    if not artist_name:
        st.info("Sélectionne un artiste dans la sidebar.")
        return

    artist = get_artist(artist_name)
    albums  = get_albums(artist_name)
    tracks  = get_tracks(artist_name)
    artist_image_url = get_artist_url(artist_name)

    if artist.empty:
        st.warning(f"Aucune donnée pour {artist_name}.")
        return
    
    artist_header(artist, artist_image_url)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    artist_kpis(artist)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Ligne 1 : Radar + Sentiment donut ─────────────────────────────────────
    col1, col2, col3 = st.columns([1.4, 1, 1], gap="small")

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Signature stylistique</div>', unsafe_allow_html=True)
        raw = {RADAR_DISPLAY[k]: safe_float(artist.get(k, 0)) for k in RADAR_KEYS if k in artist.index}
        normalized = normalize_radar(raw) if raw else {}
        if normalized:
            st.plotly_chart(radar_chart(normalized), use_container_width=True)
        else:
            st.info("Données insuffisantes.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Tonalité globale</div>', unsafe_allow_html=True)
        pos = safe_float(artist.get("avg_sentiment_positive", 0))
        neu = safe_float(artist.get("avg_sentiment_neutral",  0))
        neg = safe_float(artist.get("avg_sentiment_negative", 0))
        if pos + neu + neg > 0:
            st.plotly_chart(sentiment_donut(pos, neu, neg), use_container_width=True)
        else:
            st.info("Données sentiment absentes.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Champs lexicaux</div>', unsafe_allow_html=True)
        lex = {LEXICAL_FIELD_DISPLAY[k]: safe_float(artist.get(k, 0))
               for k in LEXICAL_FIELD_DISPLAY if k in artist.index}
        if lex and any(v > 0 for v in lex.values()):
            st.plotly_chart(lexical_bars(lex), use_container_width=True)
        else:
            st.info("Données lexicales absentes.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Ligne 2 : Top mots + Heatmap émotionnelle ─────────────────────────────
    col4, col5 = st.columns([1, 1.5], gap="small")

    with col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top 20 mots</div>', unsafe_allow_html=True)
        words = parse_top_words(artist.get("top30_words"))
        if words:
            st.plotly_chart(top_words_bar(words[:20]), use_container_width=True)
        else:
            st.info("Aucun mot disponible.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Carte émotionnelle de la discographie</div>',
                    unsafe_allow_html=True)
        if not albums.empty:
            df_heat = albums_emotion_matrix(albums)
            if not df_heat.empty:
                st.plotly_chart(emotion_heatmap(df_heat), use_container_width=True)
            else:
                st.info("Colonnes émotions absentes.")
        else:
            st.info("Aucun album disponible.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Infos stylométrie ──────────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Métriques stylométriques détaillées</div>',
                unsafe_allow_html=True)
    cols_stylo = [
        ("avg_pos_noun_ratio", "Ratio noms"),
        ("avg_pos_verb_ratio", "Ratio verbes"),
        ("avg_pos_adj_ratio",  "Ratio adj."),
        ("avg_pronoun_i_ratio","Ratio je/j'"),
        ("avg_rhyme_density",  "Densité rimes"),
        ("avg_syllables_line", "Syl./ligne"),
        ("avg_word_length",    "Long. mot moy."),
        ("avg_hapax_ratio",    "Ratio hapax"),
    ]
    c = st.columns(4, gap="small")
    for i, (col_key, col_label) in enumerate(cols_stylo):
        val = safe_float(artist.get(col_key, None))
        c[i % 4].metric(col_label, f"{val:.3f}" if val else "—")
    st.markdown('</div>', unsafe_allow_html=True)
