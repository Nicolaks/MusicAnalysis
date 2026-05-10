from __future__ import annotations
import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.loader import get_tracks, get_track, get_artists
from data.transforms import safe_float, streams_label, parse_top_words, parse_emotion_arc
from components.charts import scatter_ttr_streams, emotion_arc_line, sentiment_donut
from components.filters import artist_selector
from components.artist_header import artist_header
from data.loader import get_artist, get_albums, get_tracks, get_artist_url
from config import EMOTION_DISPLAY, COLORS


def _top_tracks_html(df: pd.DataFrame, stream_col: str) -> str:
    rows_html = ""
    for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
        name   = str(row.get("track_name", "—"))[:35]
        album  = str(row.get("album_name", ""))[:25]
        streams_val = row.get(stream_col)
        s_label = streams_label(streams_val) if pd.notna(streams_val) else "—"
        rows_html += f"""
<div class="track-row">
  <span class="track-num">{i}</span>
  <div style="flex:1;min-width:0">
    <div class="track-name">{name}</div>
    <div class="track-album">{album}</div>
  </div>
  <span class="track-streams">{s_label}</span>
</div>"""
    return rows_html


def render():
    st.title("🎵 Analyse chansons")

    artist_name = artist_selector(key="songs_artist")
    if not artist_name:
        st.info("Sélectionne un artiste.")
        return

    tracks = get_tracks(artist_name)
    if tracks.empty:
        st.warning("Aucun titre disponible.")
        return

    st.caption(f"{len(tracks)} titres · {artist_name}")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    
    artist = get_artist(artist_name)
    artist_image_url = get_artist_url(artist_name)
    artist_header(artist, artist_image_url)

    # ── Déterminer colonne streams disponible ─────────────────────────────────
    stream_col = None
    for c in ["spotify_total_streams", "streams"]:
        if c in tracks.columns and tracks[c].notna().any():
            stream_col = c
            break

    # ── Ligne 1 : Top tracks + Scatter ────────────────────────────────────────
    col1, col2 = st.columns([1, 1.5], gap="small")

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top titres par streams</div>',
                    unsafe_allow_html=True)
        if stream_col:
            top = tracks.nlargest(10, stream_col)
            st.markdown(_top_tracks_html(top, stream_col), unsafe_allow_html=True)
        else:
            top = tracks.head(10)
            st.markdown(_top_tracks_html(top, "track_name"), unsafe_allow_html=True)
            st.caption("Colonne streams absente — ordre de la base.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if stream_col and "ttr" in tracks.columns:
            st.markdown('<div class="card-title">Richesse lexicale vs Streams</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                scatter_ttr_streams(tracks, "ttr", stream_col, "TTR", "Streams"),
                use_container_width=True
            )
        elif "ttr" in tracks.columns and "rhyme_density" in tracks.columns:
            st.markdown('<div class="card-title">TTR vs densité de rimes</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                scatter_ttr_streams(tracks, "ttr", "rhyme_density",
                                     "TTR", "Densité rimes"),
                use_container_width=True
            )
        else:
            st.info("Données insuffisantes pour le scatter.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Analyse d'une chanson ─────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Analyse détaillée d\'une chanson</div>',
                unsafe_allow_html=True)

    track_names = tracks["track_name"].dropna().sort_values().tolist()
    selected    = st.selectbox("Choisir un titre", track_names, key="track_detail")

    if selected:
        row = tracks[tracks["track_name"] == selected].iloc[0]
        col_a, col_b, col_c, col_d = st.columns(4, gap="small")
        col_a.metric("Mots",       int(safe_float(row.get("word_count", 0))))
        col_b.metric("TTR",        f"{safe_float(row.get('ttr',0)):.3f}")
        col_c.metric("Rimes/ligne",f"{safe_float(row.get('rhyme_density',0)):.2f}")
        col_d.metric("Syl./ligne", f"{safe_float(row.get('avg_syllables_line',0)):.1f}")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        col_e, col_f, col_g = st.columns(3, gap="small")

        with col_e:
            pos = safe_float(row.get("sentiment_positive", 0))
            neu = safe_float(row.get("sentiment_neutral",  0))
            neg = safe_float(row.get("sentiment_negative", 0))
            if pos + neu + neg > 0:
                st.caption("Tonalité")
                st.plotly_chart(sentiment_donut(pos, neu, neg), use_container_width=True)
            else:
                st.info("Sentiment absent.")

        with col_f:
            # Émotions dominantes
            emo_cols = [f"emotion_{e}" for e in ["joie","tristesse","colere","peur","surprise","degout"]]
            emo_vals = {EMOTION_DISPLAY.get(e, e): safe_float(row.get(f"emotion_{e}", None))
                        for e in ["joie","tristesse","colere","peur","surprise","degout"]
                        if row.get(f"emotion_{e}") is not None}
            if emo_vals:
                st.caption("Profil émotionnel")
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(
                    x=list(emo_vals.values()),
                    y=list(emo_vals.keys()),
                    orientation="h",
                    marker_color=COLORS["primary"],
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=180,
                    margin=dict(l=8,r=8,t=8,b=8),
                    font=dict(family="DM Sans", size=10),
                    xaxis=dict(showgrid=False, visible=False),
                    yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Émotions absentes.")

        with col_g:
            # Arc émotionnel
            has_arc = any(row.get(f"arc_s1_{e}") is not None
                         for e in ["joie","tristesse","colere"])
            if has_arc:
                st.caption("Arc émotionnel")
                st.plotly_chart(emotion_arc_line(row), use_container_width=True)
            else:
                # Temps verbaux
                tv = {
                    "Passé":    safe_float(row.get("past_tense_ratio", None)),
                    "Présent":  safe_float(row.get("present_tense_ratio", None)),
                    "Futur":    safe_float(row.get("future_tense_ratio", None)),
                }
                if any(v > 0 for v in tv.values()):
                    st.caption("Temps verbaux")
                    import plotly.graph_objects as go
                    fig2 = go.Figure(go.Bar(
                        x=list(tv.keys()), y=list(tv.values()),
                        marker_color=COLORS["primary"],
                        text=[f"{v:.0%}" for v in tv.values()],
                        textposition="outside",
                    ))
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=180, margin=dict(l=8,r=8,t=8,b=8),
                        font=dict(family="DM Sans", size=10),
                        xaxis=dict(tickfont=dict(size=11)),
                        yaxis=dict(showgrid=False, visible=False),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Arc / temps verbaux absents.")

        # Émotion dominante
        emo_dom = row.get("emotion_dominant")
        if emo_dom and isinstance(emo_dom, str):
            cls = f"emo-{emo_dom}"
            st.markdown(
                f'Émotion dominante : <span class="{cls}">{EMOTION_DISPLAY.get(emo_dom, emo_dom)}</span>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Distribution émotions sur tout le catalogue ───────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    emo_dom_col = "emotion_dominant"
    if emo_dom_col in tracks.columns and tracks[emo_dom_col].notna().any():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Distribution des émotions dominantes sur le catalogue</div>',
                    unsafe_allow_html=True)
        counts = tracks[emo_dom_col].value_counts()
        import plotly.express as px
        fig_pie = px.pie(
            values=counts.values, names=counts.index,
            color_discrete_sequence=["#1a5c38","#185fa5","#a32d2d","#534ab7","#854f0b","#0f6e56"],
            hole=0.4,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", size=11),
            height=260,
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=8,r=8,t=8,b=8),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
