from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st
import base64
import json

sys.path.insert(0, str(Path(__file__).parent))

from components.styles import inject_global_css
from data.loader import get_corpus_stats, get_artists, get_corpus_year_range
from data.transforms import safe_float
from components.metrics import kpi_row
from config import ARTIST_DISPLAY_NAMES

from pages import (
    portrait_artiste,
    evolution_carriere,
    analyse_chansons,
    comparaison_artistes,
    nlp_audio,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music NLP Dashboard",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_pages = """
<style>
[data-testid="stSidebarNav"] {display: none;}
[data-testid="stHeader"] {display: none;}
</style>
"""
st.markdown(hide_pages, unsafe_allow_html=True)

inject_global_css()

# ── Navigation state ─────────────────────────────────────────────────────────
PAGES = {
    "dashboard":    ("🏠 Dashboard",           None),
    "portrait":     ("🎤 Portrait artiste",     portrait_artiste),
    "evolution":    ("📅 Évolution carrière",   evolution_carriere),
    "chansons":     ("🎵 Analyse chansons",     analyse_chansons),
    "comparaison":  ("📊 Comparaison artistes", comparaison_artistes),
    "nlp_audio":    ("🔊 NLP × Audio",          nlp_audio),
}

if "page" not in st.session_state:
    st.session_state.page = "dashboard"


def img_to_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()

logo_b64 = img_to_b64("dashboard/data/logo.png") 

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="margin: -4rem -1rem 1rem -1rem;">
    <img src="data:image/png;base64,{logo_b64}" 
        style="width:100%;display:block;">
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px;font-weight:600;color:#aaa;letter-spacing:.08em;padding:0 4px;margin-bottom:6px">MENU</div>', unsafe_allow_html=True)

    for key, (label, _) in PAGES.items():
        active = "active" if st.session_state.page == key else ""
        if st.sidebar.button(label, key=f"nav_{key}",
                              use_container_width=True,
                              type="secondary"):
            st.session_state.page = key
            st.rerun()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;font-weight:600;color:#aaa;letter-spacing:.08em;padding:0 4px;margin-bottom:6px">BASE DE DONNÉES</div>', unsafe_allow_html=True)

    stats = get_corpus_stats()
    st.markdown(f"""
        <div style="background:#f0f7f3;border-radius:12px;padding:12px 14px;">
        <div style="font-size:12px;color:#555;margin-bottom:6px;font-weight:500">Corpus analysé</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
            <div style="font-size:11px;color:#888">Artistes</div>
            <div style="font-size:12px;font-weight:600;color:#1a5c38;text-align:right">{stats['total_artists']}</div>
            <div style="font-size:11px;color:#888">Albums</div>
            <div style="font-size:12px;font-weight:600;color:#1a5c38;text-align:right">{stats['total_albums']}</div>
            <div style="font-size:11px;color:#888">Titres</div>
            <div style="font-size:12px;font-weight:600;color:#1a5c38;text-align:right">{stats['total_tracks']}</div>
            <div style="font-size:11px;color:#888">Mots</div>
            <div style="font-size:12px;font-weight:600;color:#1a5c38;text-align:right">{stats['total_words']:,}</div>
        </div>
        </div>
    """, unsafe_allow_html=True)


# ── Routing ───────────────────────────────────────────────────────────────────
page_key = st.session_state.page


if page_key == "dashboard":
    # ── Home dashboard ────────────────────────────────────────────────────────
    st.title("Dashboard")
    st.caption("Analyse NLP du corpus rap/ hip-hop francophone")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    
    artists = get_artists()
    if not artists:
        st.warning("⚠️ Aucun artiste trouvé. Vérifie que la base DuckDB est bien remplie.")
        st.stop()

    # ── Ligne principale : top artistes par vocabulaire + dernier pipeline ────
    from data.loader import get_artists_comparison
    import pandas as pd

    df_all = get_artists_comparison(artists)
    
    if not df_all.empty and ARTIST_DISPLAY_NAMES:
        df_all["artist_name"] = df_all["artist_name"].replace(ARTIST_DISPLAY_NAMES)

    stats = get_corpus_stats()
    years = get_corpus_year_range()
    total_streams = int(df_all["total_streams"].sum()) if not df_all.empty and "total_streams" in df_all.columns else 0
    from data.transforms import streams_label

    kpi_row([
        {"label": "Artistes analysés", "value": str(stats["total_artists"]),
        "sub": "dans le corpus", "featured": True},
        {"label": "Albums",    "value": str(stats["total_albums"]),   "sub": "discographies"},
        {"label": "Titres",    "value": str(stats["total_tracks"]),   "sub": "paroles"},
        {"label": "Mots total","value": f"{stats['total_words']:,}", "sub": "tokens filtrés"},
        {"label": "Période",   "value": f"{years['year_min']}–{years['year_max']}", "sub": "années couvertes"},
        {"label": "Streams",   "value": streams_label(total_streams), "sub": "Spotify cumulés"},
    ])

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


    col1, col3 = st.columns([1, 1], gap="small")

    with col1:
        st.markdown('<div class="card-title">Top artistes : vocabulaire carrière</div>', unsafe_allow_html=True)
        if not df_all.empty and "career_vocabulary_size" in df_all.columns:
            from components.charts import artists_compare_bar
            top_vocab = df_all.nlargest(10, "career_vocabulary_size")
            st.plotly_chart(
                artists_compare_bar(top_vocab, "career_vocabulary_size", "Mots uniques"),
                width="stretch"
            )
        else:
            st.info("Données insuffisantes.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card-title">Top artistes : TTR</div>', unsafe_allow_html=True)
        if not df_all.empty and "avg_ttr" in df_all.columns:
            from components.charts import artists_compare_bar
            top_ttr = df_all.nlargest(10, "avg_ttr")
            st.plotly_chart(
                artists_compare_bar(top_ttr, "avg_ttr", "TTR moyen"),
                width="stretch"
            )
        else:
            st.info("Données TTR absentes.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    exp_col1, exp_col2 = st.columns([0.5, 0.5])  # même ratio que col4/5/6

    with exp_col1:
        with st.expander("Vocabulaire carrière"):
            st.markdown("""
                **Vocabulaire carrière**  
                Nombre de mots utilisés sur l'ensemble de la discographie. 
                Cette métrique est sensible à la taille du catalogue : 
                - un artiste avec plus d'albums aura mécaniquement un score plus élevé. 
            """)
    
    with exp_col2:
        with st.expander("Comprendre le TTR"):
            st.markdown("""
            **TTR = Type-Token Ratio** 
            
            *Moyenne des TTR calculée chanson par chanson, indépendamment de la taille du catalogue.*  
            C'est le rapport entre le nombre de mots uniques (types) et le nombre total de mots (tokens) dans un texte. 
            
            Par exemple si un artiste utilise 1000 mots au total dont 250 mots différents → TTR = 0.25. 
            Plus le TTR est élevé, plus le vocabulaire est varié et riche. Un TTR faible indique beaucoup de répétitions. 
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    col_wc = st.container()

    with col_wc:
        st.markdown('<div class="card-title">Mots les plus fréquents : corpus global</div>', unsafe_allow_html=True)
        if not df_all.empty and "top30_words" in df_all.columns:
            from data.transforms import safe_json
            import plotly.graph_objects as go
            from collections import Counter

            word_counts = Counter()
            for val in df_all["top30_words"]:
                words = safe_json(val) if isinstance(val, str) else val
                if isinstance(words, list):
                    word_counts.update(words)

        if word_counts:
            top_words = word_counts.most_common(50)
            words, freqs = zip(*top_words)
            max_freq = max(freqs)

            # Disposition en spirale depuis le centre
            import math
            n = len(words)
            xs, ys = [], []
            for i in range(n):
                angle = i * 10  # angle doré
                radius = 0.8 * math.sqrt(i)
                xs.append(radius * math.cos(angle))
                ys.append(radius * math.sin(angle) * 0.6)  # légèrement aplati

            fig_wc = go.Figure(go.Scatter(
                x=xs,
                y=ys,
                mode="text",
                text=list(words),
                textfont=dict(
                size=[9 + (f / max_freq) * 27 for f in freqs],
                color=[
                    f"rgb({int(26 + (1 - f/max_freq) * 178)}, "
                    f"{int(92 + (1 - f/max_freq) * 149)}, "
                    f"{int(56 + (1 - f/max_freq) * 160)})"
                    for f in freqs
                ],
                family="DM Sans",
            ),
                hovertemplate=[f"<b>{w}</b><br>{f} artistes<extra></extra>"
                            for w, f in zip(words, freqs)],
            ))
            fig_wc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                margin=dict(l=8, r=8, t=8, b=8),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-6, 6]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4, 4]),
            )
            st.plotly_chart(fig_wc, width="stretch")
        
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Radar moyen corpus + activité ─────────────────────────────────────────
    col4 = st.container()

    with col4:
        st.markdown('<div class="card-title">Distribution émotions : corpus global</div>',
                    unsafe_allow_html=True)
        if not df_all.empty and "avg_emotion_scores" in df_all.columns:
            from data.transforms import safe_json
            from config import COLORS, EMOTION_DISPLAY
            import plotly.graph_objects as go

            emotion_totals = {}
            for val in df_all["avg_emotion_scores"]:
                parsed = safe_json(val) if isinstance(val, str) else val
                if parsed:
                    for k, v in parsed.items():
                        emotion_totals[k] = emotion_totals.get(k, 0) + v

            if emotion_totals:
                total = sum(emotion_totals.values())
                emo_avgs = {k: v / total for k, v in sorted(emotion_totals.items(), key=lambda x: -x[1])}

                fig_emo = go.Figure(go.Bar(
                    x=[EMOTION_DISPLAY.get(k, k) for k in emo_avgs.keys()],
                    y=list(emo_avgs.values()),
                    marker=dict(
                        color=list(emo_avgs.values()),
                        colorscale=[[0, "#4a976e"], [1, "#1a5c38"]],
                        showscale=False,
                    ),
                    text=[f"{v:.3f}" for v in emo_avgs.values()],
                    textposition="outside",
                    textfont=dict(size=9),
                ))
                fig_emo.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=240,
                    margin=dict(l=8,r=8,t=8,b=8),
                    font=dict(family="DM Sans", size=11),
                    xaxis=dict(tickfont=dict(size=11)),
                    yaxis=dict(gridcolor="#f0f0f0", tickformat=".3f"),
                )
                st.plotly_chart(fig_emo, width="stretch")
            else:
                st.info("Colonnes émotions absentes.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    col_scatter = st.container()

    with col_scatter:
        st.markdown('<div class="card-title">Corrélation streams × richesse vocabulaire (TTR)</div>', unsafe_allow_html=True)
        if not df_all.empty and "avg_ttr" in df_all.columns and "total_streams" in df_all.columns:
            import plotly.graph_objects as go
            import numpy as np

            df_sc = df_all[["artist_name", "avg_ttr", "total_streams"]].dropna()
            df_sc = df_sc[df_sc["total_streams"] > 0]
            
            
            max_freq = max(freqs) if max(freqs) > 0 else 1

            if not df_sc.empty:
                # Ligne de tendance OLS
                x = df_sc["avg_ttr"].values
                y = df_sc["total_streams"].values
                m, b = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = m * x_line + b

                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=df_sc["avg_ttr"],
                    y=df_sc["total_streams"],
                    mode="markers+text",
                    text=df_sc["artist_name"],
                    textposition="top center",
                    textfont=dict(family="DM Sans", size=11, color="#1a5c38"),
                    marker=dict(
                        size=[max(8, min(40, s / df_sc["total_streams"].max() * 50)) for s in df_sc["total_streams"]],
                        color=df_sc["avg_ttr"],
                        colorscale=[[0, "#1a5c38"], [1, "#5dbf8a"]],
                        showscale=False,
                    ),
                    hovertemplate="<b>%{text}</b><br>TTR: %{x:.3f}<br>Streams: %{y:,}<extra></extra>",
                ))
                fig_sc.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=380,
                    margin=dict(l=8, r=8, t=8, b=8),
                    font=dict(family="DM Sans", size=11),
                    xaxis=dict(title="TTR (richesse vocabulaire)", gridcolor="#f0f0f0", type="linear"),
                    yaxis=dict(title="Streams Spotify", gridcolor="#f0f0f0", tickformat=".2s"),
                    showlegend=False,
                )
                st.plotly_chart(fig_sc, width="stretch")
            else:
                st.info("Données insuffisantes.")
        else:
            st.info("Colonnes streams ou TTR absentes.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    _, module = PAGES[page_key]
    if module:
        module.render()
