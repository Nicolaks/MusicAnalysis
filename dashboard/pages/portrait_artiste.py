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
        st.plotly_chart(identity_card_chart(artist, corpus), width='stretch')

        with st.expander("Comment interpréter la carte d'identité ?"):
            st.markdown("""
            - **Mots / chanson** :  
            `≈ 50 → 2000+`  
            Volume moyen de texte par morceau. Indique la densité globale des paroles.

            - **Diversité (TTR)** :  
            *Moyenne des TTR calculée chanson par chanson, indépendamment de la taille du catalogue.*   
            `0 → 1`  
            Mesure la variété du vocabulaire. Sur 100 mots au total, si seulement 20 sont différents, 
            le TTR est de 0,20. 
            Plus ce chiffre est élevé, plus le vocabulaire est varié. 
            
            - **Rimes** :  
            `0 → +∞`  
            Fréquence moyenne des rimes par phrase. Indique le niveau de travail phonique.

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
            st.plotly_chart(lexical_bars(lex, corpus_avg=corpus.get("avg_lexical_fields", {})), width='stretch')
        else:
            st.info("Données lexicales absentes.")
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("Comment lire ce graphique ?"):
            st.markdown("""
                    **Champs lexicaux**  
                    Ce graphique décompose les paroles de l'artiste en grandes thématiques pour révéler les univers qui structurent son écriture.

                    - **Chaque barre correspond à un champ lexical** : un ensemble de mots gravitant autour d'un même thème (famille, rue, argent, violence…). 
                    
                    - **La longueur de la barre et le pourcentage associé** indiquent la part que représente ce thème dans l'ensemble du vocabulaire utilisé : plus la barre est longue, plus ce champ lexical est central dans les paroles. 
                    
                    - **La barre mise en évidence** correspond au thème dominant, celui qui pèse le plus lourd dans l'identité lyricale de l'artiste. 
                    
                    - **Les pourcentages sont calculés** sur la fréquence des mots appartenant à chaque champ lexical, rapportée au volume total de mots analysés. 
                    

                    Cette visualisation permet de cerner les **obsessions thématiques** d'un artiste : quels sont les univers qu'il convoque le plus souvent, ce qui façonne son identité et son positionnement artistique.  
                    Elle offre une lecture objective de ce dont parle vraiment un artiste, au-delà des impressions, et permet de suivre comment ces thématiques évoluent d'un album à l'autre.
                """)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Ligne 2 : Top mots + Heatmap émotionnelle ─────────────────────────────
    col4, col5, col6 = st.columns([1, 1, 1.5], gap="small")

    with col4:
        st.markdown('<div class="card-title">Top 20 mots</div>', unsafe_allow_html=True)
        words = parse_top_words(artist.get("top30_words"))
        if words:
            st.plotly_chart(top_words_bar(words[:20]), width='stretch', key="top_words_freq")
        else:
            st.info("Aucun mot disponible.")
        st.markdown('</div>', unsafe_allow_html=True)
        
# top idf avec explication

    with col5:
        st.markdown('<div class="card-title">Top mots TF-IDF</div>', unsafe_allow_html=True)
        words = parse_top_words(artist.get("tfidf_top_keywords"))
        if words:
            st.plotly_chart(top_words_bar(words[:20]), width='stretch', key="top_tfidf")
        else:
            st.info("Aucun mot disponible.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        st.markdown('<div class="card-title">Émotions globales</div>', unsafe_allow_html=True)
        emo_scores = artist.get("avg_emotion_scores")
        if emo_scores:
            st.plotly_chart(emotion_donut_chart(emo_scores), width='stretch')
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
                st.plotly_chart(emotion_heatmap(df_heat), width='stretch')
            else:
                st.info("Colonnes émotions absentes.")
        else:
            st.info("Aucun album disponible.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with st.expander("Comment lire la carte émotionnelle ?"):
        st.markdown("""
        **Évolution des émotions par album**  
        Ce graphique matriciel croise les émotions détectées avec la discographie de l'artiste pour visualiser comment sa palette émotionnelle évolue dans le temps.

        - **Chaque ligne correspond à une émotion**, chaque colonne à un album classé chronologiquement de gauche à droite.
        - **La taille de chaque bulle** traduit l'intensité de l'émotion dans l'album concerné : une grande bulle signifie que cette émotion est fortement présente dans les paroles, une petite bulle qu'elle n'est qu'effleurée.
        - **La couleur de chaque bulle** est propre à chaque émotion, ce qui permet de suivre visuellement une ligne horizontale et de repérer d'un coup d'œil les albums où elle culmine ou s'efface.
        - **Une bulle très pâle ou absente** indique que l'émotion correspondante est quasi inexistante dans cet album.

        Ce graphique est particulièrement puissant pour observer les **ruptures et continuités émotionnelles** dans une carrière.  
        Il permet de détecter des tournants artistiques. Un album où la colère cède la place à l'espoir, ou inversement, et de comprendre comment l'artiste fait évoluer sa tonalité au fil du temps, révélant ainsi une forme de **maturité ou de transformation intérieure** à travers son œuvre.
    """)
        st.markdown('</div>', unsafe_allow_html=True)
     
    # ── Star chart        ──────────────────────────────────────────────────────   
    
    st.markdown('<div class="card-title">Empreinte sonore</div>', unsafe_allow_html=True) 
    
    artist_audio, corpus_audio = get_audio_radar(artist["artist_name"])
    if not artist_audio.empty:
        st.plotly_chart(
            audio_radar_chart(artist_audio, compare_df=corpus_audio),
            width='stretch',
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
        "<ul><li><i>< 0.16 : très narratif/verbal</i></li><li><i>0.16 → 0.18 : équilibré</i></li><li><i>0.18 → 0.20 : descriptif</i></li><li><i>> 0.20 : très imagé</i></li></ul>",

        "<b>Ratio verbes</b> : intensité narrative et dynamique du discours. <br>"
        "<ul><li><i>< 0.09 : texte statique</i></li><li><i>0.09 → 0.11 : équilibré</i></li><li><i>0.11 → 0.12 : narratif / dynamique</i></li><li><i>> 0.12 : très dynamique</i></li></ul>",

        "<b>Ratio adj.</b> : richesse descriptive et précision des images. <br>"
        "<ul><li><i>< 0.045 : style brut / direct</i></li><li><i>0.045 → 0.055 : standard</i></li><li><i>0.055 → 0.075 : descriptif</i></li><li><i>> 0.075 : très visuel / littéraire</i></li></ul>",

        "<b>Ratio je/j'</b> : présence de l'artiste dans le texte. <br>"
        "<ul><li><i>< 0.015 : narration impersonnelle</i></li><li><i>0.015 → 0.035 : présence modérée</i></li><li><i>0.035 → 0.050 : introspection forte</i></li><li><i>> 0.050 : écriture très personnelle</i></li></ul>",

        "<b>Densité rimes</b> : travail de musicalité et de structure sonore. <br>"
        "<ul><li><i>< 1.0 : faible</i></li><li><i>1.0 → 1.5 : moyenne</i></li><li><i>1.5 → 2.0 : élevée</i></li><li><i>2.0 → 2.5 : très technique</i></li><li><i>> 2.5 : extrêmement dense</i></li></ul>",

        "<b>Syl./ligne</b> : complexité rythmique et densité du flow. <br><br> "
        "<ul><li><i>< 10 : flow simple / direct</i></li><li><i>10 → 14 : densité standard</i></li><li><i>14 → 17 : flow dense</i></li><li><i>> 17 : très technique / chargé</i></li></ul>",

        "<b>Long. mot moy.</b> : sophistication lexicale globale. <br>"
        "<ul><li><i>< 5.0 : langage simple / oral</i></li><li><i>5.0 → 5.5 : standard</i></li><li><i>5.5 → 5.9 : soutenu</i></li><li><i>> 5.9 : très sophistiqué</i></li></ul>",

        "<b>Ratio hapax</b> : créativité lexicale et diversité des mots. <br>"
        "<ul><li><i>< 0.60 : écriture répétitive</i></li><li><i>0.60 → 0.75 : vocabulaire varié</i></li><li><i>0.75 → 0.83 : forte richesse lexicale</i></li><li><i>> 0.83 : diversité lexicale exceptionnelle</i></li></ul>",
    ]
    
    c = st.columns(2, gap="small")
    for i, ((col_key, col_label), expl) in enumerate(zip(cols_stylo, cols_stylo_expl)):
        val = safe_float(artist.get(col_key, None))
        display_val = f"{val:.3f}" if val is not None else "—"

        c_min    = safe_float(corpus.get(f"{col_key}_min", None))
        c_max    = safe_float(corpus.get(f"{col_key}_max", None))
        c_avg    = safe_float(corpus.get(f"{col_key}_avg", None))
        c_median = safe_float(corpus.get(f"{col_key}_median", None))

        corpus_html = ""
        if all(x is not None for x in [c_min, c_max, c_avg, c_median]):
            corpus_html = f"""
            <div style='font-size:0.75em; opacity:0.6; margin-top:2px; display:flex; gap:12px; flex-wrap:wrap;'>
                <span>Moy. <b>{c_avg:.3f}</b></span>
                <span>Méd. <b>{c_median:.3f}</b></span>
                <span>Min <b>{c_min:.3f}</b></span>
                <span>Max <b>{c_max:.3f}</b></span>
            </div>
            """

        with c[i % 2]:
            st.metric(col_label, display_val)
            st.markdown(
                f"<div style='font-size:0.8em; opacity:0.7; margin-top:-8px'>{expl}</div>",
                unsafe_allow_html=True
            )
            if corpus_html:
                st.markdown(corpus_html, unsafe_allow_html=True)
            st.markdown("<div style='height:-2px'></div><hr style='opacity:0.5'>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
