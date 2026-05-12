from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from components.styles import inject_global_css
from data.loader import get_corpus_stats, get_artists
from data.transforms import safe_float

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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:4px 0 20px;">
  <div style="width:34px;height:34px;background:#1a5c38;border-radius:10px;
              display:flex;align-items:center;justify-content:center;">
    <span style="font-size:18px">🎤</span>
  </div>
  <span style="font-size:15px;font-weight:600;color:#1a1a1a;">Music NLP</span>
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
    st.caption("Analyse NLP du corpus rap français")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    stats = get_corpus_stats()

    from components.metrics import kpi_row
    kpi_row([
        {"label": "Artistes analysés", "value": str(stats["total_artists"]),
         "sub": "dans le corpus", "featured": True},
        {"label": "Albums",    "value": str(stats["total_albums"]),   "sub": "discographies"},
        {"label": "Titres",    "value": str(stats["total_tracks"]),   "sub": "paroles"},
        {"label": "Mots total","value": f"{stats['total_words']:,}", "sub": "tokens filtrés"},
    ])

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    artists = get_artists()
    if not artists:
        st.warning("⚠️ Aucun artiste trouvé. Vérifie que la base DuckDB est bien remplie.")
        st.stop()

    # ── Ligne principale : top artistes par vocabulaire + dernier pipeline ────
    from data.loader import get_artists_comparison
    import pandas as pd

    df_all = get_artists_comparison(artists)

    col1, col2, col3 = st.columns([1.5, 1, 1], gap="small")

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top artistes — vocabulaire carrière</div>',
                    unsafe_allow_html=True)
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

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Ton du corpus</div>', unsafe_allow_html=True)
        if not df_all.empty:
            avg_pos = df_all["avg_sentiment_positive"].mean() if "avg_sentiment_positive" in df_all.columns else 0
            avg_neu = df_all["avg_sentiment_neutral"].mean()  if "avg_sentiment_neutral"  in df_all.columns else 0
            avg_neg = df_all["avg_sentiment_negative"].mean() if "avg_sentiment_negative" in df_all.columns else 0
            if avg_pos + avg_neu + avg_neg > 0:
                from components.charts import sentiment_donut
                st.plotly_chart(sentiment_donut(avg_pos, avg_neu, avg_neg),
                                 width="stretch")
            else:
                st.info("Données sentiment absentes.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top artistes — TTR</div>', unsafe_allow_html=True)
        if not df_all.empty and "career_ttr" in df_all.columns:
            from components.charts import artists_compare_bar
            top_ttr = df_all.nlargest(10, "career_ttr")
            st.plotly_chart(
                artists_compare_bar(top_ttr, "career_ttr", "TTR"),
                width="stretch"
            )
        else:
            st.info("Données TTR absentes.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Radar moyen corpus + activité ─────────────────────────────────────────
    col4, col5 = st.columns([1.2, 1], gap="small")

    with col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Distribution émotions — corpus global</div>',
                    unsafe_allow_html=True)
        if not df_all.empty:
            from config import EMOTION_LABELS, EMOTION_DISPLAY
            import plotly.graph_objects as go
            emo_avgs = {}
            for e in EMOTION_LABELS:
                col_name = f"avg_emotion_{e}"
                if col_name in df_all.columns:
                    emo_avgs[EMOTION_DISPLAY[e]] = float(df_all[col_name].mean())
            if emo_avgs:
                palette = ["#1a5c38","#185fa5","#a32d2d","#534ab7","#854f0b","#0f6e56"]
                fig_emo = go.Figure(go.Bar(
                    x=list(emo_avgs.keys()),
                    y=list(emo_avgs.values()),
                    marker_color=palette,
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

    with col5:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Accès rapide</div>', unsafe_allow_html=True)
        shortcuts = [
            ("portrait",    "🎤 Portrait artiste",     "Explorer la signature d'un artiste"),
            ("evolution",   "📅 Évolution carrière",   "Voir les tendances album par album"),
            ("chansons",    "🎵 Analyse chansons",     "Plonger dans une chanson"),
            ("comparaison", "📊 Comparaison",          "Comparer plusieurs artistes"),
            ("nlp_audio",   "🔊 NLP × Audio",          "Croiser paroles et son"),
        ]
        for key, label, desc in shortcuts:
            st.markdown(f"""
<div onclick="" style="padding:8px 10px;border-radius:10px;border:1px solid #e8ede9;
     margin-bottom:6px;cursor:pointer;background:#fafcfa;">
  <div style="font-size:12px;font-weight:600;color:#1a5c38">{label}</div>
  <div style="font-size:11px;color:#aaa">{desc}</div>
</div>""", unsafe_allow_html=True)
            if st.button(f"→ {label}", key=f"shortcut_{key}", width="content"):
                st.session_state.page = key
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

else:
    _, module = PAGES[page_key]
    if module:
        module.render()
