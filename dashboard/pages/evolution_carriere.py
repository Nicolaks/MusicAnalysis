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
        
        with st.expander("Que nous dit ce graphique ?"):
            st.markdown("""
                **Évolution des 4 émotions dominantes par album**  
                Ce graphique suit dans le temps l'intensité des quatre émotions les plus caractéristiques de l'artiste, album après album.

                - **Chaque ligne représente une émotion**, identifiable par sa couleur dans la légende en bas du graphique.
                - **L'axe horizontal** liste les albums dans l'ordre chronologique, de la première à la dernière sortie.
                - **L'axe vertical** indique le score moyen de chaque émotion pour l'album correspondant : plus la ligne est haute, plus cette émotion est intensément présente dans les paroles.
                - **Les croisements et écarts entre les lignes** sont particulièrement révélateurs : quand les courbes se resserrent, les émotions s'équilibrent ; quand elles s'écartent, une émotion prend clairement le dessus sur les autres.
                - **Les chutes ou pics brutaux** sur un album signalent un changement de registre fort, potentiellement lié à un événement de vie ou une rupture artistique volontaire.

                C'est l'un des graphiques les plus riches pour comprendre la **trajectoire émotionnelle** d'un artiste sur l'ensemble de sa carrière : il permet de voir si la tonalité s'assombrit ou s'éclaircit avec le temps, si certaines émotions disparaissent progressivement, et si des cycles se répètent d'un projet à l'autre.
            """)
        
    col2 = st.container()

    with col2:
        st.markdown('<div class="card-title">Évolution de la richesse lexicale au fil des albums</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(vocab_evolution(albums), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    st.info("💡 Glissez la barre ci-dessus pour naviguer dans la discographie.")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    
    with st.expander("Comment lire cette analyse ?"):
        st.markdown("""
                **Évolution de la richesse lexicale au fil des albums**  
                Ce graphique suit deux indicateurs complémentaires sur l'ensemble de la discographie pour mesurer l'évolution du vocabulaire de l'artiste.

                - **La courbe foncée: Vocabulaire album** (axe gauche) indique le nombre total de mots distincts utilisés dans chaque album. Elle est sensible à la taille du projet : un album avec plus de titres aura mécaniquement un score plus élevé.
                - **La courbe claire: TTR** (axe droit) mesure le *Type-Token Ratio*, c'est-à-dire le rapport entre les mots uniques et le nombre total de mots. Cet indicateur est indépendant de la longueur de l'album : un TTR élevé signifie que l'artiste répète peu ses mots et diversifie son expression.
                - **Les deux axes sont distincts** : il ne faut pas comparer les valeurs absolues des deux courbes, mais observer leurs tendances respectives et leurs éventuels décalages.
                - **La barre de navigation en bas** permet de zoomer sur une période précise de la discographie pour affiner la lecture.

                L'intérêt de croiser ces deux métriques est de distinguer la **quantité** de vocabulaire de sa **qualité** : un artiste peut écrire beaucoup tout en se répétant, ou au contraire produire des projets courts mais lexicalement très denses. Les écarts entre les deux courbes révèlent ces nuances et permettent de repérer les albums où l'écriture gagne/perd en richesse réelle.
            """)

    # ── Heatmap émotions ──────────────────────────────────────────────────────
    colEmo = st.container()
    
    with colEmo:
        st.markdown('<div class="card-title">Barre émotionnelle (8 émotions principales) : album par album</div>',
                    unsafe_allow_html=True)
        has_emo = "avg_emotion_scores" in albums.columns and albums["avg_emotion_scores"].notna().any()
        if has_emo:
            st.plotly_chart(emotion_stacked_bars(albums), width='stretch')
        else:
            st.info("Colonnes émotions absentes.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        
        with st.expander("Décryptage de cette visualisation"):
            st.markdown("""
                **Barre émotionnelle des 8 émotions principales par album**  
                Ce graphique en barres empilées donne une photographie complète de la composition émotionnelle de chaque album.

                - **Chaque barre représente un album**, classé chronologiquement de gauche à droite, et atteint toujours 100% : elle ne mesure pas l'intensité absolue des émotions, mais leur **poids relatif les unes par rapport aux autres** au sein de ce projet.
                - **Chaque segment coloré** correspond à une émotion identifiable via la légende en bas. Sa hauteur indique la part qu'elle occupe dans l'ensemble émotionnel de l'album.
                - **La lecture verticale** d'une barre permet de saisir en un instant la palette émotionnelle d'un album : est-il dominé par une seule émotion ou réparti équitablement entre plusieurs ?
                - **La lecture horizontale**, en comparant les barres entre elles, permet de repérer les glissements de tonalité d'un projet à l'autre : un segment qui grandit ou rétrécit au fil du temps trahit une évolution dans l'écriture.

                Ce graphique est complémentaire aux courbes d'évolution : là où ces dernières montrent l'intensité, la barre émotionnelle révèle les **équilibres internes** de chaque album. Elle permet de comprendre si un artiste s'ancre dans un registre émotionnel stable ou si chaque projet propose une configuration affective différente, signe d'une écriture en constante transformation.
            """)

    # ── Champs lexicaux + Lisibilité ──────────────────────────────────────────
    col3 = st.container()

    with col3:
        st.markdown('<div class="card-title">Évolution des champs lexicaux</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(lexical_area(albums), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    
    with st.expander("Mode d'emploi de ce graphique"):
        st.markdown("""
                **Évolution des champs lexicaux par album**  
                Ce graphique en aires empilées retrace la présence de chaque grande thématique dans les paroles, album après album.

                - **Chaque couche colorée correspond à un champ lexical**, identifiable via la légende en bas. L'ordre d'empilement est constant d'un album à l'autre, ce qui facilite la comparaison visuelle dans le temps.
                - **L'épaisseur de chaque couche** reflète le poids de ce champ lexical dans l'album : une couche qui s'élargit signifie que la thématique prend plus de place dans les paroles, une couche qui se réduit indique qu'elle s'efface.
                - **La hauteur totale de la pile** varie d'un album à l'autre et reflète la densité thématique globale du projet : un pic vers le haut signale un album particulièrement chargé en références à ces univers.
                - **Les ondulations de la surface** sont le signe de fluctuations thématiques : l'écriture n'est jamais figée, certains sujets reviennent par vagues, d'autres s'installent progressivement ou disparaissent.

                Ce graphique permet de lire la **trajectoire narrative** d'un artiste sur le long terme. Il révèle si certains thèmes sont des constantes de son écriture ou des passages, si des ruptures thématiques coïncident avec des moments charnières de sa carrière, et comment l'ensemble de ses préoccupations se réorganise d'un projet à l'autre.
            """)

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
