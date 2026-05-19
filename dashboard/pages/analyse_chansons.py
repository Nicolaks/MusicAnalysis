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
        st.markdown('<div class="card-title">Albums les plus streamés</div>', unsafe_allow_html=True)
        if "album_name" in tracks.columns and stream_col:
            album_streams = (
                tracks.groupby("album_name")[stream_col]
                .sum()
                .reset_index(name="total_streams")
                .sort_values("total_streams", ascending=True)
                .tail(10)
            )
            n = len(album_streams)
            height_album = 500 if n >= 10 else max(250, n * 32)
            import plotly.graph_objects as go
            fig_albums = go.Figure(go.Bar(
                x=album_streams["total_streams"],
                y=album_streams["album_name"].apply(lambda x: x[:20] + "…" if len(str(x)) > 20 else str(x)),
                orientation="h",
                marker=dict(
                    color=album_streams["total_streams"],
                    colorscale=[[0, "#e8f5ee"], [1, "#1a5c38"]],
                    showscale=False,
                ),
                text=album_streams["total_streams"].apply(streams_label),
                textposition="auto",
                textfont=dict(size=10),
                hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>",
            ))
            fig_albums.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=height_album,
                margin=dict(l=8, r=50, t=8, b=8),
                font=dict(family="DM Sans", size=10),
                xaxis=dict(showgrid=False, visible=False),
                yaxis=dict(tickfont=dict(size=10)),
            )
            st.plotly_chart(fig_albums, width="stretch")
        else:
            st.info("Données streams absentes.")
        st.markdown('</div>', unsafe_allow_html=True)

    col3 = st.container()

    with col3:
        if stream_col and "ttr" in tracks.columns:
            st.markdown('<div class="card-title">Richesse lexicale vs Streams</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                scatter_ttr_streams(tracks, "ttr", stream_col, "TTR", "Streams"),
                width='stretch'
            )
        elif "ttr" in tracks.columns and "rhyme_density" in tracks.columns:
            st.markdown('<div class="card-title">TTR vs densité de rimes</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                scatter_ttr_streams(tracks, "ttr", "rhyme_density",
                                     "TTR", "Densité rimes"),
                width='stretch'
            )
        else:
            st.info("Données insuffisantes pour le scatter.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Analyse d'une chanson ─────────────────────────────────────────────────
    st.markdown('<div class="card-title">Analyse détaillée d\'une chanson</div>',
                unsafe_allow_html=True)

    track_names = tracks[tracks["artist_name"] == artist_name]["track_name"].dropna().sort_values().tolist()
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
            # Sentiment absent de la table — on affiche les émotions positives/négatives
            import json
            emo_raw = row.get("emotion_scores")
            if emo_raw:
                try:
                    emo_parsed = json.loads(emo_raw) if isinstance(emo_raw, str) else emo_raw
                    # Regroupe en positif/négatif/neutre
                            
                    pos_keys = {"espoir","gratitude", "envie", "embarras", "nostalgie", "surprise", "sympathie", "amour", "joie", "culpabilité"}
                    neg_keys = {"méfiance", "désespoir","mépris","indignation", "jalousie", "honte", "dégoût", "peur", "colère", "tristesse"}
                    pos = sum(v for k, v in emo_parsed.items() if k in pos_keys)
                    neg = sum(v for k, v in emo_parsed.items() if k in neg_keys)
                    neu = max(0, 1 - pos - neg)
                    st.caption("Tonalité")
                    st.plotly_chart(sentiment_donut(pos, neu, neg), width="stretch")
                except Exception:
                    st.info("Sentiment absent.")
            else:
                st.info("Sentiment absent.")

        with col_f:
            import json
            emo_raw = row.get("emotion_scores")
            if emo_raw:
                try:
                    emo_parsed = json.loads(emo_raw) if isinstance(emo_raw, str) else emo_raw
                    emo_vals = dict(sorted(
                        {k: v for k, v in emo_parsed.items() if v > 0}.items(),
                        key=lambda x: x[1], reverse=True
                    )[:6])

                    if emo_vals:
                        st.caption("Profil émotionnel")
                        import plotly.graph_objects as go
                        total = sum(emo_vals.values())
                        n = len(emo_vals)

                        def gradient_green(i, total):
                            t = i / max(total - 1, 1)
                            r = int(0x1a + (0xe8 - 0x1a) * t)
                            g = int(0x5c + (0xf5 - 0x5c) * t)
                            b = int(0x38 + (0xee - 0x38) * t)
                            return f"rgb({r},{g},{b})"

                        fig = go.Figure()
                        for i, (emo, val) in enumerate(emo_vals.items()):
                            pct = val / total * 100
                            color = gradient_green(i, n)
                            fig.add_trace(go.Bar(
                                x=[""],
                                y=[pct],
                                width=[0.3],
                                name=emo.capitalize(),
                                marker_color=color,
                                hovertemplate=f"<b>{emo.capitalize()}</b> : {pct:.1f}%<extra></extra>",
                            ))
                            
                        fig.update_layout(
                            barmode="stack",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=250,
                            margin=dict(l=8, r=8, t=8, b=8),
                            font=dict(family="DM Sans", size=10),
                            xaxis=dict(showgrid=False, visible=False),
                            yaxis=dict(showgrid=False, visible=False, range=[0, 100]),
                            legend=dict(orientation="v", x=1.02, font=dict(size=9)),
                            showlegend=True,
                        )
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("Émotions absentes.")
                except Exception:
                    st.info("Émotions absentes.")
            else:
                st.info("Émotions absentes.")

        with col_g:
            import json
            LEXICAL_COLORS = {
                "argent":       "rgba(194,153,70,0.6)",
                "rue":          "rgba(26,92,56,0.6)",
                "famille":      "rgba(59,156,161,0.6)",
                "drogue":       "rgba(83,74,183,0.6)",
                "célébrité":    "rgba(207,131,92,0.6)",
                "spiritualité": "rgba(15,110,86,0.6)",
                "amour_perdu":  "rgba(201,104,122,0.6)",
                "violence":     "rgba(163,45,45,0.6)",
                "succès":       "rgba(46,138,87,0.6)",
                "échec":        "rgba(79,93,117,0.6)",
                "liberté":      "rgba(112,130,56,0.6)",
                "prison":       "rgba(125,81,104,0.6)",
                "mort":         "rgba(28,40,54,0.6)",
                "fête":         "rgba(218,159,166,0.6)",
                "sport":        "rgba(69,123,157,0.6)",
                "mode":         "rgba(218,159,166,0.6)",
                "voitures":     "rgba(107,114,92,0.6)",
            }
            FALLBACK = [
                "rgba(244,162,97,0.6)",
                "rgba(231,111,81,0.6)",
                "rgba(38,70,83,0.6)",
                "rgba(42,157,143,0.6)",
                "rgba(233,196,106,0.6)",
                "rgba(168,218,220,0.6)",
                "rgba(69,123,157,0.6)",
                "rgba(230,57,70,0.6)",
            ]

            lex_raw = row.get("lexical_field_scores")
            if lex_raw:
                try:
                    lex_parsed = json.loads(lex_raw) if isinstance(lex_raw, str) else lex_raw
                    lex_vals = dict(sorted(
                        {k: v for k, v in lex_parsed.items() if v > 0}.items(),
                        key=lambda x: x[1], reverse=True
                    )[:6])

                    if lex_vals:
                        st.caption("Champs lexicaux")
                        import plotly.graph_objects as go
                        labels = [k.replace("_", " ").capitalize() for k in lex_vals.keys()]
                        values = list(lex_vals.values())
                        colors = [LEXICAL_COLORS.get(k, FALLBACK[i % len(FALLBACK)])
                                  for i, k in enumerate(lex_vals.keys())]

                        fig3 = go.Figure(go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.6,
                            marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                            textinfo="none",
                            hovertemplate="<b>%{label}</b> : %{percent}<extra></extra>",
                            sort=False,
                        ))
                        dominant = labels[0] if labels else ""
                        fig3.update_layout(
                            annotations=[dict(
                                text=f"<b>{dominant}</b>",
                                x=0.5, y=0.5,
                                font=dict(family="DM Sans", size=11, color="#444"),
                                showarrow=False,
                            )],
                            paper_bgcolor="rgba(0,0,0,0)",
                            height=250,
                            margin=dict(l=8, r=8, t=8, b=8),
                            font=dict(family="DM Sans", size=10),
                            legend=dict(orientation="v", x=1.02, font=dict(size=9)),
                            showlegend=True,
                        )
                        st.plotly_chart(fig3, width="stretch")
                    else:
                        st.info("Champs lexicaux absents.")
                except Exception:
                    st.info("Champs lexicaux absents.")
            else:
                st.info("Champs lexicaux absents.")
