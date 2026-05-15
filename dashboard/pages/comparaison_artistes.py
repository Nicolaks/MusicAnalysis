from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.loader import get_artists_comparison
from data.transforms import safe_float, normalize_radar
from components.charts import artists_compare_bar, _LAYOUT
from components.artist_header import artist_header
from data.loader import get_artist, get_albums, get_tracks, get_artist_url
from components.filters import artist_selector
from components.filters import multi_artist_selector, metric_selector
from config import (NLP_FEATURES, NLP_FEATURES_DISPLAY, RADAR_KEYS,
                    RADAR_DISPLAY, COLORS, EMOTION_LABELS, EMOTION_DISPLAY)



def _multi_radar(df: pd.DataFrame) -> go.Figure:
    palette = ["#1a5c38", "#185fa5", "#a32d2d", "#534ab7", "#854f0b", "#0f6e56"]
    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        raw   = {RADAR_DISPLAY[k]: safe_float(row.get(k, 0)) for k in RADAR_KEYS if k in row.index}
        normd = normalize_radar(raw) if raw else {}
        if not normd:
            continue
        cats  = list(normd.keys()) + [list(normd.keys())[0]]
        vals  = list(normd.values()) + [list(normd.values())[0]]
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, name=row.get("artist_name", f"Artiste {i}"),
            fill="toself",
            line=dict(color=color, width=2),
            fillcolor=color + "25",
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=9), gridcolor="#eee"),
            angularaxis=dict(tickfont=dict(size=10, color="#555")),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
        **_LAYOUT, height=320,
    )
    return fig


def _emotion_grouped_bar(df: pd.DataFrame) -> go.Figure:
    palette = ["#1a5c38","#185fa5","#a32d2d","#534ab7","#854f0b","#0f6e56"]
    emo_cols = [f"avg_emotion_{e}" for e in EMOTION_LABELS]
    avail    = [c for c in emo_cols if c in df.columns]
    if not avail:
        return go.Figure()
    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        fig.add_trace(go.Bar(
            name=row.get("artist_name", f"Artiste {i}"),
            x=[EMOTION_DISPLAY.get(c.replace("avg_emotion_",""), c) for c in avail],
            y=[safe_float(row.get(c, 0)) for c in avail],
            marker_color=palette[i % len(palette)],
        ))
    fig.update_layout(
        **_LAYOUT, height=280, barmode="group",
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#f0f0f0", tickformat=".3f"),
        legend=dict(orientation="h", y=-0.3, font=dict(size=10)),
    )
    return fig


def render():
    st.title("📊 Comparaison artistes")

    artist_names = multi_artist_selector(key="comp_artists", default_n=3)
    if not artist_names:
        st.info("Sélectionne au moins un artiste.")
        return

    df = get_artists_comparison(artist_names)
    if df.empty:
        st.warning("Aucune donnée disponible pour ces artistes.")
        return
    
    artist_name = artist_selector(key="portrait_artist")
    artist = get_artist(artist_name)
    artist_image_url = get_artist_url(artist_name)
    artist_header(artist, artist_image_url)

    st.caption(f"{len(df)} artistes comparés")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── KPI grid artistes ──────────────────────────────────────────────────────
    st.markdown('<div class="card-title">Vue d\'ensemble</div>', unsafe_allow_html=True)
    header_cols = ["artist_name", "track_count", "album_count",
                   "career_vocabulary_size", "career_ttr",
                   "avg_rhyme_density", "pct_positive", "pct_negative"]
    avail_cols  = [c for c in header_cols if c in df.columns]
    rename_map  = {
        "artist_name":            "Artiste",
        "track_count":            "Titres",
        "album_count":            "Albums",
        "career_vocabulary_size": "Vocabulaire",
        "career_ttr":             "TTR",
        "avg_rhyme_density":      "Densité rimes",
        "pct_positive":           "% Positif",
        "pct_negative":           "% Négatif",
    }
    shown = df[avail_cols].rename(columns=rename_map)
    st.dataframe(shown, width='stretch', hide_index=True,
                 column_config={
                     "TTR":          st.column_config.NumberColumn(format="%.3f"),
                     "Densité rimes":st.column_config.NumberColumn(format="%.3f"),
                     "% Positif":    st.column_config.NumberColumn(format="%.1%"),
                     "% Négatif":    st.column_config.NumberColumn(format="%.1%"),
                 })
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Radar multi-artistes ───────────────────────────────────────────────────
    col1, col2 = st.columns([1.2, 1], gap="small")

    with col1:
        st.markdown('<div class="card-title">Signature stylistique comparée</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_multi_radar(df), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-title">Émotions dominantes comparées</div>',
                    unsafe_allow_html=True)
        fig_emo = _emotion_grouped_bar(df)
        if fig_emo.data:
            st.plotly_chart(fig_emo, width='stretch')
        else:
            st.info("Colonnes émotions absentes.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Bar chart par métrique ─────────────────────────────────────────────────
    st.markdown('<div class="card-title">Classement par métrique NLP</div>',
                unsafe_allow_html=True)

    avail_metrics = [m for m in NLP_FEATURES if m in df.columns]
    if avail_metrics:
        metric = metric_selector(avail_metrics, NLP_FEATURES_DISPLAY,
                                  label="Métrique", key="comp_metric")
        label  = NLP_FEATURES_DISPLAY.get(metric, metric)
        st.plotly_chart(artists_compare_bar(df, metric, label), width='stretch')
    else:
        st.info("Aucune métrique NLP disponible.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Scatter positionnement ─────────────────────────────────────────────────
    st.markdown('<div class="card-title">Positionnement artistique</div>',
                unsafe_allow_html=True)

    scatter_opts = [m for m in NLP_FEATURES if m in df.columns]
    if len(scatter_opts) >= 2:
        sc1, sc2 = st.columns(2, gap="small")
        with sc1:
            x_metric = metric_selector(scatter_opts, NLP_FEATURES_DISPLAY,
                                        label="Axe X", key="scatter_x")
        with sc2:
            y_default = scatter_opts[1] if scatter_opts[1] != x_metric else scatter_opts[0]
            y_metric  = metric_selector(scatter_opts, NLP_FEATURES_DISPLAY,
                                         label="Axe Y", key="scatter_y")

        palette = ["#1a5c38","#185fa5","#a32d2d","#534ab7","#854f0b","#0f6e56"]
        fig_sc = go.Figure()
        for i, (_, row) in enumerate(df.iterrows()):
            fig_sc.add_trace(go.Scatter(
                x=[safe_float(row.get(x_metric, 0))],
                y=[safe_float(row.get(y_metric, 0))],
                mode="markers+text",
                name=row.get("artist_name",""),
                text=[row.get("artist_name","")],
                textposition="top center",
                textfont=dict(size=11),
                marker=dict(size=16, color=palette[i % len(palette)]),
                showlegend=False,
            ))
        fig_sc.update_layout(
            **_LAYOUT, height=300,
            xaxis=dict(title=NLP_FEATURES_DISPLAY.get(x_metric, x_metric), gridcolor="#f0f0f0"),
            yaxis=dict(title=NLP_FEATURES_DISPLAY.get(y_metric, y_metric), gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig_sc, width='stretch')
    else:
        st.info("Métriques insuffisantes pour le scatter.")
    st.markdown('</div>', unsafe_allow_html=True)
