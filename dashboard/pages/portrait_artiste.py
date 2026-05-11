from __future__ import annotations
import streamlit as st
import sys, os
import ast
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.transforms import (parse_top_words, parse_style_signature,
                              normalize_radar, albums_emotion_matrix,
                              safe_float, streams_label)
from components.charts import (radar_chart, top_words_bar, emotion_heatmap,
                                sentiment_donut, lexical_bars, emotion_donut_chart, identity_card_chart)
from data.loader import get_artist, get_albums, get_tracks, get_artist_url, get_albums_with_streams, get_corpus_stats
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
    albums = get_albums(artist_name)
    albums_streams = get_albums_with_streams(artist_name)
    tracks  = get_tracks(artist_name)
    artist_image_url = get_artist_url(artist_name)
    corpus = get_corpus_stats()

    if artist.empty:
        st.warning(f"Aucune donnée pour {artist_name}.")
        return
    
    artist_header(artist, artist_image_url)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    artist_kpis(artist)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Ligne 1 : Radar + Sentiment donut ─────────────────────────────────────
    col1, col2 = st.columns([1.4, 2], gap="small")

    with col1:
        st.markdown('<div class="card-title">Carte d\'identité</div>', unsafe_allow_html=True)
        st.plotly_chart(identity_card_chart(artist, corpus), use_container_width=True)
        with st.expander("Comprendre les métriques"):
            st.markdown("""
            - **Diversité lexicale (TTR)** :  
            `0 → 1`  
            Plus la valeur est élevée, plus le vocabulaire est varié.

            - **Densité de rimes** :  
            `0 → 1`  
            Mesure la fréquence des schémas de rimes dans les paroles.

            - **Complexité syntaxique** :  
            Valeur généralement comprise entre `5 → 25`  
            Basée sur la longueur et la structure des phrases.

            - **Répétition** :  
            `0 → 1`  
            Plus la valeur est élevée, plus les paroles utilisent des répétitions.

            - **Charge émotionnelle** :  
            `-100 → +100`  
            Négatif = tonalité sombre  
            Positif = tonalité lumineuse.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-title">Champs lexicaux</div>', unsafe_allow_html=True)
        lex_raw = artist.get("avg_lexical_field_scores")
        try:
            lex_parsed = json.loads(lex_raw) if isinstance(lex_raw, str) else {}
            lex = {LEXICAL_FIELD_DISPLAY.get(k, k): safe_float(v)
                for k, v in lex_parsed.items()}
        except Exception:
            lex = {}
        if lex and any(v > 0 for v in lex.values()):
            st.plotly_chart(lexical_bars(lex), use_container_width=True)
        else:
            st.info("Données lexicales absentes.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Ligne 2 : Top mots + Heatmap émotionnelle ─────────────────────────────
    col4, col5, col6 = st.columns([1, 1, 1.5], gap="small")

    with col4:
        st.markdown('<div class="card-title">Top 20 mots</div>', unsafe_allow_html=True)
        words = parse_top_words(artist.get("top30_words"))
        if words:
            st.plotly_chart(top_words_bar(words[:20]), use_container_width=True)
        else:
            st.info("Aucun mot disponible.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="card-title">Émotions globales</div>', unsafe_allow_html=True)
        emo_scores = artist.get("avg_emotion_scores")
        if emo_scores:
            st.plotly_chart(emotion_donut_chart(emo_scores), use_container_width=True)
            # Émotion dominante en texte sous le chart
            dominant_raw = artist.get("dominant_emotions", "")
            if dominant_raw:
                try:
                    dominant_list = ast.literal_eval(dominant_raw)
                    dominant = ", ".join(dominant_list)
                except:
                    dominant = str(dominant_raw)
                st.caption(f"Émotion dominante · **{dominant}**")
        else:
            st.info("Données émotions absentes.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="card-title">Carte émotionnelle de la discographie</div>', unsafe_allow_html=True)
        if not albums.empty:
            df_heat = albums_emotion_matrix(albums_streams.head(10))
            if not df_heat.empty:
                st.plotly_chart(emotion_heatmap(df_heat), use_container_width=True)
            else:
                st.info("Colonnes émotions absentes.")
        else:
            st.info("Aucun album disponible.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Infos stylométrie ──────────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
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
