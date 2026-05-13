from __future__ import annotations
import streamlit as st
import sys, os
import ast
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.transforms import (parse_top_words, parse_style_signature,
                              normalize_radar, albums_emotion_matrix,
                              safe_float, streams_label)
from components.charts import (top_words_bar, emotion_heatmap,
                                sentiment_donut, lexical_bars, emotion_donut_chart, identity_card_chart, audio_radar_chart)
from data.loader import get_artist, get_albums, get_tracks, get_artist_url, get_albums_with_streams, get_corpus_stats, get_audio_radar, get_streams_artist
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
    stream = get_streams_artist(artist_name)

    if artist.empty:
        st.warning(f"Aucune donnée pour {artist_name}.")
        return
    
    artist_header(artist, artist_image_url)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    artist_kpis(artist, stream)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Ligne 1 : Radar + Sentiment donut ─────────────────────────────────────
    col1, col2 = st.columns([1.4, 2], gap="small")

    with col1:
        st.markdown('<div class="card-title">Carte d\'identité</div>', unsafe_allow_html=True)
        st.plotly_chart(identity_card_chart(artist, corpus), use_container_width=True)
        with st.expander("Comprendre les métriques"):
            st.markdown("""
            - **Mots / chanson** :  
            `≈ 50 → 2000+`  
            Volume moyen de texte par morceau. Indique la densité globale des paroles.

            - **Diversité (TTR)** :  
            `0 → 1`  
            Mesure la variété du vocabulaire. Plus c’est élevé, plus l’artiste utilise de mots différents.

            - **Rimes** :  
            `0 → 1`  
            Fréquence des schémas de rimes dans les paroles. Indique le niveau de travail phonique.

            - **Richesse vocab. (hapax)** :  
            `0 → 1`  
            Proportion de mots utilisés une seule fois. Mesure la créativité lexicale et le renouvellement du vocabulaire.

            - **Auto-référence** :  
            `0 → 1`  
            Proportion de “je/j’”. Indique le degré d’introspection et de narration personnelle.

            - **Répétition** :  
            `0 → 1`  
            Mesure la redondance des mots ou phrases. Élevé = refrains / motifs répétés / hook-centric.

            - **Complexité mots** :  
            `≈ 3 → 8+ lettres`  
            Longueur moyenne des mots. Indique la sophistication lexicale et le niveau de langage.
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
            st.plotly_chart(lexical_bars(lex, corpus_avg=corpus.get("avg_lexical_fields", {})), use_container_width=True)
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
            st.plotly_chart(top_words_bar(words[:20]), use_container_width=True, key="top_words_freq")
        else:
            st.info("Aucun mot disponible.")
        st.markdown('</div>', unsafe_allow_html=True)
        
# top idf avec explication

    with col5:
        st.markdown('<div class="card-title">Top mots TF-IDF</div>', unsafe_allow_html=True)
        words = parse_top_words(artist.get("tfidf_top_keywords"))
        if words:
            st.plotly_chart(top_words_bar(words[:20]), use_container_width=True, key="top_tfidf")
        else:
            st.info("Aucun mot disponible.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
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
                    
    exp_col1, exp_col2 = st.columns([0.55, 0.45])  # même ratio que col4/5/6

    with exp_col1:
        with st.expander("Comprendre le TOP 20 VS TF-IDF"):
            st.markdown("""
            **Top 20 mots** : Le Top 20 des mots utilisés affiche les termes les plus fréquents dans les paroles d’un artiste.
            Il permet d’identifier rapidement les thèmes dominants, les habitudes d’écriture et le champ lexical principal de son univers musical. 
            Cette analyse met en avant le vocabulaire le plus présent, mais pas forcément le plus distinctif.

            **TF-IDF** : Le Top TF-IDF met en évidence les mots les plus caractéristiques d’un artiste par rapport au reste du corpus. 
            Contrairement au simple comptage de fréquence, cette méthode valorise les termes fréquemment utilisés par l’artiste mais rares chez les autres. 
            Elle permet ainsi d’identifier les éléments qui rendent son style et son univers lexical uniques.
            """)
    
    with exp_col2:
        with st.expander("Comprendre l’analyse émotionnelle"):
            st.markdown("""
            Cette visualisation représente la répartition moyenne des émotions détectées dans les paroles de l’artiste.

            ### Émotions analysées
            - **Joie** : énergie positive, optimisme
            - **Tristesse** : mélancolie, douleur émotionnelle
            - **Colère** : agressivité, tension
            - **Peur** : anxiété, insécurité
            - **Surprise** : intensité émotionnelle soudaine
            - **Dégoût** : rejet, mépris

            ### Lecture du graphique
            - Plus une section est grande, plus cette émotion est dominante
            - Les scores représentent une intensité émotionnelle moyenne sur l’ensemble du catalogue analysé
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card-title">Carte émotionnelle de la discographie</div>', unsafe_allow_html=True)
        if not albums.empty:
            df_heat = albums_emotion_matrix(albums_streams.head(15))
            if not df_heat.empty:
                st.plotly_chart(emotion_heatmap(df_heat), use_container_width=True)
            else:
                st.info("Colonnes émotions absentes.")
        else:
            st.info("Aucun album disponible.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with st.expander("Comment lire la carte émotionnelle ?"):
        st.markdown("""
                    ### Carte émotionnelle de la discographie

                    La carte émotionnelle représente l’intensité des émotions détectées dans les paroles de chaque album. 
                    Les émotions (joie, tristesse, colère, peur, surprise, dégoût) sont affichées sur l’axe vertical et les albums sur l’axe horizontal.  
                    Plus la taille de la bulle est importante, plus l’émotion est présente dans l’album concerné.  
                    Cette visualisation permet d’identifier rapidement les émotions dominantes, les variations d’ambiance et l’évolution émotionnelle de l’artiste au fil de sa carrière.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
     
    # ── Star chart        ──────────────────────────────────────────────────────   
    
    st.markdown('<div class="card-title">Empreinte sonore</div>', unsafe_allow_html=True) 
    
    artist_audio, corpus_audio = get_audio_radar(artist["artist_name"])
    if not artist_audio.empty:
        st.plotly_chart(
            audio_radar_chart(artist_audio, compare_df=corpus_audio),
            use_container_width=True,
            key="audio_radar"
        )
    else:
        st.info("Pas de données audio disponibles.")
    with st.expander("Comment lire l'empreinte sonore ?"):
        st.markdown("""
        Ce radar représente la **signature acoustique moyenne** de l'artiste, 
        calculée sur l'ensemble de ses extraits audio disponibles.
        
        **Les 6 dimensions :**
        - **Rapidité** — tempo moyen en BPM. Un score élevé = sons rapides, urgents.
        - **Puissance** — force des beats détectés. -> Élevé = rythmique marquée et percussive.
        - **Brillance** — ratio d'énergie dans les hautes fréquences. -> Élevé = son aérien, crisp.
        - **Chaleur** — ratio d'énergie dans les basses fréquences. -> Élevé = son chaud, grave.
        - **Rugosité** — irrégularité spectrale. -> Élevé = saturation, distorsion, agressivité sonore.
        - **Flow** — densité des attaques par seconde. -> Élevé = débit syllabique rapide et dense.
        
        **La zone verte claire** autour de la toile principale représente la variabilité 
        (écart-type), plus elle est large sur un axe, plus l'artiste est **inconsistant** 
        sur cette dimension selon les morceaux.
        
        **La toile grise pointillée** est la moyenne du corpus entier,
        elle permet de situer l'artiste par rapport aux autres.
        """)
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
    
    cols_stylo_expl = [
        "<b>Ratio noms</b> : structure plus descriptive et statique du texte (univers, objets, scènes). <br>"
        " <ul><li><i> < 0.12 : très narratif/verbal</i></li> <li> <i> 0.12 → 0.20 : équilibré</i> </li> <li><i> 0.20 → 0.28 : descriptif</i></li> <li><i> > 0.28 : très imagé</i></li></ul>",

        "<b>Ratio verbes</b> : intensité narrative et dynamique du discours. <br>"
        " <ul><li><i>< 0.08 : texte statique</i></li><li><i>0.08 → 0.14 : équilibré</i></li><li><i>0.14 → 0.20 : narratif / dynamique</i></li><li><i>> 0.20 : très dynamique</i></li></ul>",

        "<b>Ratio adj.</b> : richesse descriptive et précision des images. <br>"
        " <ul><li><i>< 0.03 : style brut / direct</i></li><li><i>0.03 → 0.07 : standard</i></li><li><i>0.07 → 0.12 : descriptif</i></li><li><i>> 0.12 : très visuel / littéraire</i></li></ul>",

        "<b>Ratio je/j’</b> : présence de l’artiste dans le texte. <br>"
        " <ul><li><i>< 0.03 : narration impersonnelle</i></li><li><i>0.03 → 0.08 : présence modérée</i></li><li><i>0.08 → 0.15 : introspection forte</i></li><li><i>> 0.15 : écriture très personnelle</i></li></ul>",

        "<b>Densité rimes</b> : travail de musicalité et de structure sonore. <br>"
       " <ul><li><i>< 0.8 : faible</i></li><li><i>0.8 → 1.3 : moyenne</i></li><li><i>1.3 → 1.8 : élevée</i></li><li><i>1.8 → 2.5 : très technique</i></li><li><i>> 2.5 : extrêmement dense</i></li></ul>",

        "<b>Syl./ligne</b> : complexité rythmique et densité du flow. <br>"
        " <ul><li><i>< 8 : flow simple / direct</i></li><li><i>8 → 12 : densité standard</i></li><li><i>12 → 16 : flow dense</i></li><li><i>> 16 : très technique / chargé</i></li><br></ul>",

        "<b>Long. mot moy.</b> : sophistication lexicale globale. <br>"
        " <ul><li><i>< 4.3 : langage simple / oral</i></li><li><i>4.3 → 5.0 : standard</i></li><li><i>5.0 → 5.6 : soutenu</i></li><li><i>> 5.6 : très sophistiqué</i></li></ul>",

        "<b>Ratio hapax</b> : créativité lexicale et diversité des mots. <br>"
        " <ul><li><i>< 0.45 : écriture répétitive</i></li><li><i>0.45 → 0.65 : vocabulaire varié</i></li><li><i>0.65 → 0.80 : forte richesse lexicale</i></li><li><i>> 0.80 : diversité lexicale exceptionnelle</i></li></ul>",
    ]
    
    c = st.columns(2, gap="small")
    for i, ((col_key, col_label), expl) in enumerate(zip(cols_stylo, cols_stylo_expl)):
        val = safe_float(artist.get(col_key, None))
        display_val = f"{val:.3f}" if val is not None else "—"

        with c[i % 2]:
            st.metric(col_label, display_val)
            st.markdown(
                f"<div style='font-size:0.8em; opacity:0.7; margin-top:-8px'>{expl}</div>",
                unsafe_allow_html=True
            )
            st.markdown("<div style='height:-2px'></div><hr style='opacity:0.5'>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
