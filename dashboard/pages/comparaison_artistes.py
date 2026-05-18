from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os
import json
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.loader import get_artists_comparison
from data.transforms import safe_float, normalize_radar
from components.charts import artists_compare_bar, _LAYOUT
from components.artist_header import artist_header
from data.loader import get_artist, get_albums, get_tracks, get_artist_url, get_embeddings_all_artists
from components.filters import artist_selector
from components.filters import multi_artist_selector, metric_selector
from config import (NLP_FEATURES, NLP_FEATURES_DISPLAY, RADAR_KEYS,
                    RADAR_DISPLAY, COLORS, EMOTION_LABELS, EMOTION_DISPLAY)
from sklearn.decomposition import PCA
import plotly.graph_objects as go



def _multi_radar(df: pd.DataFrame) -> go.Figure:
    palette = [
        ("rgba(26,92,56,1)",     "rgba(26,92,56,0.15)"),
        ("rgba(24,95,165,1)",    "rgba(24,95,165,0.15)"),
        ("rgba(163,45,45,1)",    "rgba(163,45,45,0.15)"),
        ("rgba(83,74,183,1)",    "rgba(83,74,183,0.15)"),
        ("rgba(133,79,11,1)",    "rgba(133,79,11,0.15)"),
        ("rgba(15,110,86,1)",    "rgba(15,110,86,0.15)"),
        ("rgba(201,104,122,1)",  "rgba(201,104,122,0.15)"),
        ("rgba(207,131,92,1)",   "rgba(207,131,92,0.15)"),
        ("rgba(59,156,161,1)",   "rgba(59,156,161,0.15)"),
        ("rgba(125,81,104,1)",   "rgba(125,81,104,0.15)"),
        ("rgba(112,130,56,1)",   "rgba(112,130,56,0.15)"),
        ("rgba(194,153,70,1)",   "rgba(194,153,70,0.15)"),
        ("rgba(79,93,117,1)",    "rgba(79,93,117,0.15)"),
        ("rgba(189,58,58,1)",    "rgba(189,58,58,0.15)"),
        ("rgba(90,107,124,1)",   "rgba(90,107,124,0.15)"),
        ("rgba(107,114,92,1)",   "rgba(107,114,92,0.15)"),
        ("rgba(244,162,97,1)",   "rgba(244,162,97,0.15)"),
        ("rgba(42,157,143,1)",   "rgba(42,157,143,0.15)"),
        ("rgba(69,123,157,1)",   "rgba(69,123,157,0.15)"),
        ("rgba(218,159,166,1)",  "rgba(218,159,166,0.15)"),
    ]
    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        raw   = {RADAR_DISPLAY[k]: safe_float(row.get(k, 0)) for k in RADAR_KEYS if k in row.index}
        normd = normalize_radar(raw) if raw else {}
        if not normd:
            continue
        cats  = list(normd.keys()) + [list(normd.keys())[0]]
        vals  = list(normd.values()) + [list(normd.values())[0]]
        line_color, fill_color = palette[i % len(palette)]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, name=row.get("artist_name", f"Artiste {i}"),
            fill="toself",
            line=dict(color=line_color, width=2),
            fillcolor=fill_color,
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

def centroid_chart(names: list[str], embs: np.ndarray, selected: list[str] = None) -> go.Figure:
    selected = selected or []
    
    pca = PCA(n_components=3)
    coords = pca.fit_transform(embs)

    palette = ["#1a5c38", "#185fa5", "#a32d2d", "#534ab7", "#854f0b", "#0f6e56",
               "#c9687a", "#cf835c", "#3b9ca1", "#708238"]

    fig = go.Figure()

    # Artistes non sélectionnés — quasi invisibles pour garder le contexte spatial
    mask_others = np.array([n not in selected for n in names])
    fig.add_trace(go.Scatter3d(
        x=coords[mask_others, 0],
        y=coords[mask_others, 1],
        z=coords[mask_others, 2],
        mode="markers",
        text=[n for n, m in zip(names, mask_others) if m],
        hovertemplate="<b>%{text}</b><extra></extra>",
        marker=dict(size=4, color="rgba(200,200,200,0.04)", line=dict(width=0)),
        showlegend=False,
    ))

    # Artistes sélectionnés mis en valeur
    for i, artist in enumerate(selected):
        if artist not in names:
            continue
        idx = names.index(artist)
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter3d(
            x=[coords[idx, 0]],
            y=[coords[idx, 1]],
            z=[coords[idx, 2]],
            mode="markers+text",
            name=artist,
            text=[artist],
            textposition="top center",
            textfont=dict(size=10, family="DM Sans", color=color),
            hovertemplate=f"<b>{artist}</b><extra></extra>",
            marker=dict(size=10, color=color, line=dict(color="#ffffff", width=1.5)),
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(210,225,245,0.5)",
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title=dict(text="Composante 1", font=dict(size=10, color="#aaa")),
                showticklabels=False,
                gridcolor="rgba(180,200,230,0.5)",
                backgroundcolor="rgb(220,232,248)",
                showbackground=True,
                linecolor="rgba(150,180,220,0.8)",
                linewidth=2,
                showline=True,
            ),
            yaxis=dict(
                title=dict(text="Composante 2", font=dict(size=10, color="#aaa")),
                showticklabels=False,
                gridcolor="rgba(180,200,230,0.5)",
                backgroundcolor="rgb(220,232,248)",
                showbackground=True,
                linecolor="rgba(150,180,220,0.8)",
                linewidth=2,
                showline=True,
            ),
            zaxis=dict(
                title=dict(text="Composante 3", font=dict(size=10, color="#aaa")),
                showticklabels=False,
                gridcolor="rgba(180,200,230,0.5)",
                backgroundcolor="rgb(220,232,248)",
                showbackground=True,
                linecolor="rgba(150,180,220,0.8)",
                linewidth=2,
                showline=True,
            ),
        ),
        font=dict(family="DM Sans", size=10),
        height=700,
        margin=dict(l=0, r=0, t=0, b=40),
        legend=dict(orientation="h", y=-0.05, font=dict(size=10)),
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
    
    col3 = st.container()
    
    with col3:
        st.markdown('<div class="card-title">Comparaison 3D des artistes</div>', unsafe_allow_html=True)
        
        rows  = get_embeddings_all_artists()
        names = [r[0] for r in rows]
        embs  = np.array([json.loads(r[1]) for r in rows])

        fig_centroid = centroid_chart(names, embs, selected=artist_names)
        
        if fig_centroid.data:
            st.plotly_chart(fig_centroid, use_container_width=True, key="Centroid 3D")
        else:
            st.error("Informations indisponibles pour les artistes sélectionnés")
            
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(220,232,248,0.4) 0%, rgba(232,245,238,0.4) 100%);
                border-radius: 12px;
                padding: 20px 24px;
                margin-top: 12px;
                border-left: 3px solid rgba(130,165,210,0.6);
                font-family: 'DM Sans', sans-serif;
            ">
                <div style="font-size: 0.95em; font-weight: 600; color: #1a5c38; margin-bottom: 10px;">
                    🔭 Comment lire ce graphique ?
                </div>
                <div style="font-size: 0.85em; color: #555; line-height: 1.7;">
                    Chaque point représente un artiste, positionné dans un espace à 3 dimensions 
                    calculé à partir de l'ensemble de ses paroles. <b>Plus deux artistes sont proches, 
                    plus leur univers lyrical se ressemble</b> dans les thèmes abordés, le vocabulaire 
                    utilisé et le style d'écriture.<br><br>
                    Les trois axes (<i>Composante 1, 2, 3</i>) sont des directions mathématiques abstraites 
                    qui capturent les principales différences stylistiques entre artistes. Ils n'ont pas 
                    de nom fixe, ils émergent naturellement des données.<br><br>
                    <span style="color: #888; font-size: 0.9em;">
                    💡 Astuce : faites tourner le graphique en cliquant-glissant pour explorer 
                    les regroupements sous différents angles.
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

