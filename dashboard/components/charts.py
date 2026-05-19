from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
import numpy as np
import streamlit as st
from data.transforms import safe_float, normalize_radar
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (COLORS, EMOTION_LABELS, EMOTION_DISPLAY, RADAR_KEYS,RADAR_DISPLAY, COLORS, EMOTION_LABELS,
                    EMOTION_DISPLAY, LEXICAL_COLORS, FALLBACK, RADAR_AUDIO_RANGES, RADAR_AUDIO_KEYS, FALLBACK_EMOTION_HEATMAP,
                    EMOTION_COLORS_HEATMAP, EMOTION_COLORS_RGBA, PALETTE_RADAR_MULTI_ARTISTS, COLORS_STREAM_TTR_MULTI)
from sklearn.decomposition import PCA

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#444", size=11),
    margin=dict(l=8, r=8, t=30, b=8),
)

def _apply(fig: go.Figure, height: int = 260, title: str = "") -> go.Figure:
    fig.update_layout(**_LAYOUT, height=height, title=dict(
        text=title, font=dict(size=13, color="#1a1a1a"), x=0, pad=dict(l=0)
    ) if title else None)
    return fig

def normalize_radar_audio(values: dict) -> dict:
    """Normalisation sur plages réelles plutôt que min/max local."""
    result = {}
    for k, v in values.items():
        mn, mx = RADAR_AUDIO_RANGES.get(k, (0, 1))
        if mx == mn:
            result[k] = 0.5
        else:
            result[k] = float(np.clip((v - mn) / (mx - mn), 0, 1))
    return result


def audio_radar_chart(artist_df: pd.DataFrame, compare_df: pd.DataFrame | None = None) -> go.Figure:
    if artist_df.empty:
        return go.Figure()

    keys = list(RADAR_AUDIO_KEYS.keys())
    available = [k for k in keys if k in artist_df.columns]
    if not available:
        return go.Figure()

    # ── Moyennes et écarts-type de l'artiste ──
    means = {k: float(artist_df[k].mean()) for k in available}
    stds  = {k: float(artist_df[k].std(ddof=0)) for k in available}

    norm_means     = normalize_radar_audio(means)
    norm_means_std = normalize_radar_audio({k: means[k] + stds[k] for k in available})
    norm_means_std = {k: min(v, 1.0) for k, v in norm_means_std.items()}  # clip à 1

    labeled       = {RADAR_AUDIO_KEYS[k]: norm_means[k]     for k in available}
    labeled_upper = {RADAR_AUDIO_KEYS[k]: norm_means_std[k] for k in available}

    cats       = list(labeled.keys())       + [list(labeled.keys())[0]]
    vals       = list(labeled.values())     + [list(labeled.values())[0]]
    vals_upper = list(labeled_upper.values()) + [list(labeled_upper.values())[0]]

    fig = go.Figure()

    # Zone std (halo autour de la moyenne)
    fig.add_trace(go.Scatterpolar(
        r=vals_upper, theta=cats,
        fill="toself",
        fillcolor="rgba(26,92,56,0.08)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Moyenne artiste
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats,
        fill="toself", name="Artiste",
        line=dict(color=COLORS["primary"], width=2),
        fillcolor="rgba(26,92,56,0.18)",
        hovertemplate="<b>%{theta}</b><br>Score : %{r:.2f}<extra></extra>",
    ))

    # ── Corpus moyen (optionnel) ──
    if compare_df is not None and not compare_df.empty:
        c_means      = {k: float(compare_df[k].mean()) for k in available}
        c_normalized = normalize_radar_audio(c_means)
        c_labeled    = {RADAR_AUDIO_KEYS[k]: c_normalized[k] for k in available}
        c_cats = list(c_labeled.keys())   + [list(c_labeled.keys())[0]]
        c_vals = list(c_labeled.values()) + [list(c_labeled.values())[0]]

        fig.add_trace(go.Scatterpolar(
            r=c_vals, theta=c_cats,
            fill="toself", name="Corpus moyen",
            line=dict(color="#888780", width=1.5, dash="dot"),
            fillcolor="rgba(136,135,128,0.10)",
            hovertemplate="<b>%{theta}</b><br>Corpus : %{r:.2f}<extra></extra>",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont=dict(size=12), gridcolor="#eee",
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["25%", "50%", "75%", "100%"],
            ),
            angularaxis=dict(tickfont=dict(size=11, color="#555")),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, font=dict(size=13)),
        **{**_LAYOUT, "margin": dict(t=10, b=40, l=40, r=40)},
        height=400,
    )
    return fig


# ── Sentiment line ───────────────────────────────────────────────────────────

def sentiment_line(df: pd.DataFrame | None, x_col: str = "album_name") -> go.Figure:
    fig = go.Figure()
    
    pairs = [
        ("avg_sentiment_positive", "Positif",  COLORS["positive"]),
        ("avg_sentiment_neutral",  "Neutre",   COLORS["neutral"]),
        ("avg_sentiment_negative", "Négatif",  COLORS["negative"]),
    ]
    for col, label, color in pairs:
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[col], name=label,
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
        ))
    fig.update_layout(
        **_LAYOUT, height=260,
        xaxis=dict(tickangle=-30, gridcolor="#f0f0f0", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#f0f0f0", tickformat=".2f"),
        legend=dict(orientation="h", y=-0.3, font=dict(size=10)),
    )
    return fig


# ── Emotion heatmap ──────────────────────────────────────────────────────────

def emotion_heatmap(df: pd.DataFrame) -> go.Figure:
    cols_present = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not cols_present or df.empty:
        return go.Figure()

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def interpolate_color(base_hex: str, t: float) -> str:
        """t=0 → très clair, t=1 → couleur pleine."""
        r, g, b = hex_to_rgb(base_hex)
        # Mélange avec blanc (255,255,255) selon t
        r2 = int(255 + (r - 255) * t)
        g2 = int(255 + (g - 255) * t)
        b2 = int(255 + (b - 255) * t)
        return f"rgb({r2},{g2},{b2})"

    means = df[cols_present].mean()
    cols_present = means.sort_values(ascending=False).head(9).index.tolist()

    albums = [a[:15] + "…" if len(str(a)) > 15 else str(a) for a in df.index.tolist()]
    emotions = [EMOTION_DISPLAY.get(c, c.capitalize()) for c in cols_present]

    fig = go.Figure()

    for i, (col, emo_label) in enumerate(zip(cols_present, emotions)):
        base_color = EMOTION_COLORS_HEATMAP.get(col, FALLBACK[i % len(FALLBACK)])
        vals = df[col].fillna(0).values
        vmax = vals.max() if vals.max() > 0 else 1

        # t : 0 = valeur faible (clair), 1 = valeur max (foncé)
        t_vals = [(v / vmax) ** 0.6 for v in vals]
        colors = [interpolate_color(base_color, t) for t in t_vals]
        sizes  = [3 + t * 19 for t in t_vals]

        fig.add_trace(go.Scatter(
            x=albums,
            y=[emo_label] * len(albums),
            mode="markers",
            name=emo_label,
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.9,
                line=dict(color="#ffffff", width=1.5),
            ),
            text=[f"<b>{emo_label}</b><br>{v:.3f}" for v in vals],
            hovertemplate="%{x}<br>%{text}<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT,
        height=max(300, 55 * len(cols_present) + 80),
        showlegend=False,
        xaxis=dict(tickangle=-35, tickfont=dict(size=13, color="#888"), showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=15, color="#555"), showgrid=True, gridcolor="#f0f0f0", gridwidth=1, zeroline=False),
    )
    return fig

# ── Lexical fields bars ──────────────────────────────────────────────────────

def lexical_bars(values: dict[str, float], corpus_avg: dict[str, float] = {}) -> go.Figure:
    labels = list(values.keys())
    vals   = list(values.values())
    total  = sum(vals) if sum(vals) > 0 else 1
    pcts   = [v / total * 100 for v in vals]
    colors = [COLORS["primary"] if v == max(vals) else COLORS["primary_light"] for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors,
        textposition="none",
        showlegend=False,
    ))

    for i, pct in enumerate(pcts):
        fig.add_annotation(
            x=max(vals) * 1.18, y=i,
            text=f"{pct:.1f}%",
            showarrow=False,
            font=dict(size=12, color="#555", family="DM Sans"),
            xanchor="right",
            yanchor="middle",
        )

    fig.update_layout(
        **_LAYOUT,
        height=max(300, 32 * len(labels) + 60),
        xaxis=dict(range=[0, max(vals) * 1.25 if vals else 1], showgrid=False, visible=False),
        yaxis=dict(tickfont=dict(size=13), autorange="reversed"),
        bargap=0.2,
        showlegend=False,
    )
    return fig

def identity_card_chart(artist: pd.Series, corpus: pd.Series) -> go.Figure:

    metrics = [
        ("Mots / chanson",  "avg_word_count",      "wc_min",    "wc_max",    "wc_avg",    "{:.0f}"),
        ("Diversité (TTR)", "avg_ttr",              "ttr_min",   "ttr_max",   "ttr_avg",   "{:.2f}"),
        ("Rimes",           "avg_rhyme_density",    "rhyme_min", "rhyme_max", "rhyme_avg", "{:.2f}"),
        ("Richesse vocab.", "avg_hapax_ratio",      "hapax_min", "hapax_max", "hapax_avg", "{:.2f}"),
        ("Auto-référence",  "avg_pronoun_i_ratio",  "i_min",     "i_max",     "i_avg",     "{:.2f}"),
        ("Répétition",      "avg_repetition_ratio", "rep_min",   "rep_max",   "rep_avg",   "{:.2f}"),
        ("Complexité mots", "avg_word_length",      "wl_min",    "wl_max",    "wl_avg",    "{:.1f}"),
    ]

    rows = []
    for label, key, cmin, cmax, cavg, fmt in metrics:
        val = safe_float(artist.get(key))
        if val is None:
            continue
        vmin = corpus.get(cmin, 0)
        vmax = corpus.get(cmax, 1)
        avg  = corpus.get(cavg, (vmin + vmax) / 2)
        # Normalise tout en 0-1 pour comparer sur même axe
        def norm(v): return max(0.0, min(1.0, (v - vmin) / (vmax - vmin))) if vmax != vmin else 0.5
        rows.append({
            "label": label,
            "val":   val,
            "norm":  norm(val),
            "avg":   norm(avg),
            "fmt":   fmt.format(val),
            "above": norm(val) >= norm(avg),
        })

    if not rows:
        return go.Figure()

    labels = [r["label"] for r in rows]
    norms  = [r["norm"]  for r in rows]
    avgs   = [r["avg"]   for r in rows]
    texts  = [r["fmt"]   for r in rows]
    colors = [COLORS["primary"] if r["above"] else COLORS["primary_light"] for r in rows]

    fig = go.Figure()

    # Barre fond gris
    fig.add_trace(go.Bar(
        x=[1.0] * len(rows),
        y=labels,
        orientation="h",
        marker=dict(color="#f0f0f0", line=dict(width=0)),
        showlegend=False,
        hoverinfo="skip",
        width=0.5,
    ))
    # Barre artiste
    fig.add_trace(go.Bar(
        x=norms,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        showlegend=False,
        hovertemplate="%{y} : %{text}<extra></extra>",
        text=texts,
        textposition="none",
        width=0.5,
    ))

    # Texte toujours après x=1.05
    for i, r in enumerate(rows):
        fig.add_annotation(
            x=1.05, y=i,
            text=r["fmt"],
            showarrow=False,
            font=dict(size=11, color="#555", family="DM Sans"),
            xanchor="left",
            yanchor="middle",
        )

    # Ligne verticale moyenne corpus
    for i, r in enumerate(rows):
        fig.add_shape(
            type="line",
            x0=r["avg"], x1=r["avg"],
            y0=i - 0.3,
            y1=i + 0.3,
            line=dict(color="#949494", width=1),
            layer="above",
        )

    # Légende
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="#949494", width=2),
        name="Moyenne corpus",
        showlegend=True,
    ))

    fig.update_layout(
        **_LAYOUT,
        barmode="overlay",
        height=max(300, 52 * len(rows) + 80),
        bargap=0.3,
        xaxis=dict(visible=False, range=[0, 1.5]),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="left",
            x=0,
            font=dict(size=10, family="DM Sans", color="#555"),
        ),
        showlegend=True,
    )
    return fig

# ── Top words bars ───────────────────────────────────────────────────────────

def top_words_bar(words: list[str], counts: list[int] | None = None) -> go.Figure:
    words = words[:20]
    y     = counts[:20] if counts else list(range(len(words), 0, -1))
    fig = go.Figure(go.Bar(
        x=y, y=words, orientation="h",
        marker=dict(
            color=y,
            colorscale=[[0, "#e8f5ee"], [1, "#1a5c38"]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        **_LAYOUT, height=380,
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
    )
    return fig


# ── Scatter TTR vs streams ────────────────────────────────────────────────────

def scatter_ttr_streams(df: pd.DataFrame, x_col: str, y_col: str,
                         x_label: str, y_label: str,
                         color_col: str | None = None) -> go.Figure:
    df2 = df[[x_col, y_col, "track_name"]].dropna()
    if df2.empty:
        return go.Figure()
    fig = px.scatter(
        df2, x=x_col, y=y_col, hover_name="track_name",
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig.update_traces(marker=dict(size=7, opacity=0.7, color=COLORS["primary"]))
    fig.update_layout(
        **_LAYOUT, height=400,
        xaxis=dict(title=x_label, gridcolor="#f0f0f0"),
        yaxis=dict(title=y_label, gridcolor="#f0f0f0"),
    )
    return fig


# ── Correlation heatmap ───────────────────────────────────────────────────────

def correlation_heatmap(df: pd.DataFrame, cols: list[str], labels: dict[str, str]) -> go.Figure:
    sub  = df[cols].dropna()
    if sub.empty or len(sub) < 3:
        return go.Figure()
    corr = sub.corr().round(2)
    tick_labels = [labels.get(c, c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=tick_labels, y=tick_labels,
        colorscale=[[0, "#a32d2d"], [0.5, "#f5f5f5"], [1, "#1a5c38"]],
        zmid=0,
        text=corr.values,
        texttemplate="%{text:.2f}",
        textfont=dict(size=8),
        colorbar=dict(thickness=10, tickfont=dict(size=9)),
    ))
    fig.update_layout(
        **_LAYOUT, height=450,
        xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


# ── Sentiment donut ───────────────────────────────────────────────────────────

def sentiment_donut(pos: float, neu: float, neg: float) -> go.Figure:
    fig = go.Figure(go.Pie(
        values=[pos, neu, neg],
        labels=["Positif", "Neutre", "Négatif"],
        hole=0.65,
        marker=dict(colors=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]]),
        textfont=dict(size=10),
        showlegend=True,
    ))
    layout = _LAYOUT.copy()
    layout["margin"] = dict(l=8, r=8, t=8, b=8)

    fig.update_layout(
        **layout,
        height=250,
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        annotations=[dict(
            text=f"{pos:.0%}",
            x=0.5,
            y=0.5,
            font=dict(
                size=18,
                color=COLORS["positive"],
                family="DM Sans"
            ),
            showarrow=False,
        )],
    )
    return fig


# ── Emotion arc line ─────────────────────────────────────────────────────────

def emotion_arc_line(row: pd.Series) -> go.Figure:
    segments = ["Intro", "Développement", "Outro"]
    fig = go.Figure()
    for emo in EMOTION_LABELS:
        vals = [row.get(f"arc_s{i}_{emo}", None) for i in range(1, 4)]
        if all(v is None for v in vals):
            continue
        vals = [v if v is not None else 0 for v in vals]
        fig.add_trace(go.Scatter(
            x=segments, y=vals,
            name=EMOTION_DISPLAY.get(emo, emo),
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=8),
        ))
    fig.update_layout(
        **_LAYOUT, height=240,
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0", tickformat=".2f"),
        legend=dict(orientation="h", y=-0.3, font=dict(size=10)),
    )
    return fig


# ── Bar chart comparaison artistes ───────────────────────────────────────────

def artists_compare_bar(df: pd.DataFrame, metric: str, label: str) -> go.Figure:
    sub = df[["artist_name", metric]].dropna().sort_values(metric, ascending=True)
    if sub.empty:
        return go.Figure()
    colors = [COLORS["primary"] if i == len(sub) - 1 else COLORS["primary_light"]
              for i in range(len(sub))]
    max_val = sub[metric].max()
    texts_position = ["inside" if v > max_val * 0.7 else "outside" for v in sub[metric]]
    texts_color = ["white" if v > max_val * 0.7 else "#555" for v in sub[metric]]

    fig = go.Figure(go.Bar(
        x=sub[metric], y=sub["artist_name"], orientation="h",
        marker_color=colors,
        text=sub[metric].round(3),
        textposition=texts_position,
        textfont=dict(size=10, color=texts_color),
    ))
    fig.update_layout(
        **_LAYOUT, height=max(220, 40 * len(sub) + 90),
        xaxis=dict(showgrid=False, visible=False, title=label),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig

# ── Vocab evolution line ─────────────────────────────────────────────────────

def vocab_evolution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    pairs = [
        ("album_vocabulary_size", "Vocabulaire album", COLORS["primary"]),
        ("album_ttr",             "TTR",               COLORS["primary_light"]),
    ]
    df2 = df.sort_values("release_year", na_position="last").reset_index(drop=True)
    x   = df2["album_name"]

    for col, label, color in pairs:
        if col not in df2.columns:
            continue
        yax = "y2" if col == "album_ttr" else "y"
        fig.add_trace(go.Scatter(
            x=x, y=df2[col], name=label,
            mode="lines+markers", yaxis=yax,
            line=dict(color=color, width=2.5, shape="spline", smoothing=0.8),
            marker=dict(size=7, color=color),
            customdata=df2["release_year"],
            hovertemplate="<b>%{x}</b><br>%{customdata}<br>Nombre de mots : %{y:,.0f}<extra></extra>" if col != "album_ttr"
                     else "<b>%{x}</b><br>%{customdata}<br>TTR : %{y:.3f}<extra></extra>",
        ))

    n_albums = len(df2)
    use_slider = n_albums > 8

    xaxis_cfg = dict(
        tickangle=-30,
        gridcolor="#f0f0f0",
        tickfont=dict(size=10),
    )
    if use_slider:
        xaxis_cfg["rangeslider"] = dict(visible=True, thickness=0.08)
        xaxis_cfg["range"] = [0, min(7, n_albums - 1)]

    fig.update_layout(
        **_LAYOUT, height=400 if use_slider else 400,
        xaxis=xaxis_cfg,
        yaxis=dict(gridcolor="#f0f0f0", title="Vocabulaire", tickfont=dict(size=10)),
        yaxis2=dict(overlaying="y", side="right", title="TTR",
                    tickformat=".3f", tickfont=dict(size=10)),
        legend=dict(orientation="h", y=4, font=dict(size=10)),
    )
    return fig

def emotion_donut_chart(avg_emotion_scores: str | None) -> go.Figure:
    if not avg_emotion_scores:
        return go.Figure()
    
    scores = json.loads(avg_emotion_scores)
    

    scores = {k: v for k, v in scores.items() if v > 0}
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    scores = dict(list(scores.items())[:14])

    n = len(scores)
    labels = [EMOTION_DISPLAY.get(k, k.capitalize()) for k in scores.keys()]
    values = list(scores.values())
    dominant = labels[0] if labels else ""
    
    def interpolate_green(i, total):
        t = i / max(total - 1, 1)
        r = int(166 + (26 - 166) * t)   # 166 → 26
        g = int(210 + (92 - 210) * t)   # 210 → 92
        b = int(140 + (56 - 140) * t)   # 140 → 56
        return f"rgb({r},{g},{b})"

    colors = [interpolate_green(i, n) for i in range(n)]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="none",
        hovertemplate="<b>%{label}</b> : %{percent}<extra></extra>",
        sort=False,
    ))

    fig.update_layout(
        annotations=[dict(
            text=f"<b>{dominant}</b>",
            x=0.5, y=0.5,
            font=dict(family="DM Sans", size=13, color="#444"),
            showarrow=False,
        )],
        showlegend=False,
        **_LAYOUT,
        height=280,
    )
    return fig

def emotion_lines(df: pd.DataFrame, x_col: str = "album_name") -> go.Figure: 
    records = []
    for _, row in df.iterrows():
        raw = row.get("avg_emotion_scores")
        parsed = {}
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception:
                pass
        parsed[x_col] = row[x_col]
        records.append(parsed)

    emo_df = pd.DataFrame(records).set_index(x_col)
    emo_cols = [c for c in emo_df.columns if c != x_col]

    if not emo_cols:
        return go.Figure()

    global_means = emo_df[emo_cols].astype(float).mean()
    top4 = global_means.nlargest(4).index.tolist()

    fig = go.Figure()
    for i, col in enumerate(top4):
        color = EMOTION_COLORS_HEATMAP.get(col, FALLBACK_EMOTION_HEATMAP[i % len(FALLBACK_EMOTION_HEATMAP)])
        fig.add_trace(go.Scatter(
            x=emo_df.index,
            y=emo_df[col].astype(float),
            name=col.capitalize(),
            mode="lines+markers",
            line=dict(color=color, width=2.5, shape="spline", smoothing=0.8),
            marker=dict(size=7, color=color),
            hovertemplate=f"<b>%{{x}}</b><br>{col.capitalize()} : %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT, height=450,
        xaxis=dict(tickangle=-30, gridcolor="#f0f0f0", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#f0f0f0", tickformat=".3f"),
        legend=dict(orientation="h", y=-0.35, font=dict(size=10)),
    )
    return fig

def emotion_stacked_bars(df: pd.DataFrame, x_col: str = "album_name") -> go.Figure:
    """Répartition émotionnelle par album en barres empilées 100%."""   
    records = []
    for _, row in df.iterrows():
        raw = row.get("avg_emotion_scores")
        parsed = {}
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception:
                pass
        parsed[x_col] = row[x_col]
        records.append(parsed)

    emo_df = pd.DataFrame(records).set_index(x_col)
    emo_cols = [c for c in emo_df.columns if c != x_col]

    if not emo_cols:
        return go.Figure()

    global_means = emo_df[emo_cols].astype(float).mean()
    top_cols = global_means.nlargest(8).index.tolist()

    # Normalise chaque album à 100%
    emo_df = emo_df[top_cols].astype(float)
    totals = emo_df.sum(axis=1)
    emo_pct = emo_df.div(totals, axis=0) * 100
    
    labels = [a[:18] + "…" if len(str(a)) > 12 else str(a) for a in emo_df.index]

    fig = go.Figure()
    fallback_i = 0
    for col in top_cols:
        if col in EMOTION_COLORS_RGBA:
            color = EMOTION_COLORS_RGBA[col]
        else:
            color = FALLBACK[fallback_i % len(FALLBACK)]
            fallback_i += 1

        fig.add_trace(go.Bar(
            name=col.capitalize(),
            x=emo_pct.index,
            y=emo_pct[col],
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{col.capitalize()} : %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT,
        barmode="stack",
        height=550,
        xaxis=dict(
                tickangle=-30,
                gridcolor="#f0f0f0",
                tickfont=dict(size=10),
                tickmode="array",
                tickvals=list(emo_df.index),
                ticktext=labels,
            ),
        yaxis=dict(ticksuffix="%", gridcolor="#f0f0f0", tickfont=dict(size=10), range=[0, 100]),
        legend=dict(orientation="h", y=-0.35, font=dict(size=10)),
    )
    return fig

def lexical_area(df):
    if "avg_lexical_field_scores" not in df.columns:
        return go.Figure()

    records = []
    for _, row in df.iterrows():
        raw = row.get("avg_lexical_field_scores")
        parsed = {}
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception:
                pass
        parsed["album_name"] = row["album_name"]
        records.append(parsed)

    lex_df = pd.DataFrame(records).set_index("album_name")
    lex_cols = [c for c in lex_df.columns if c != "album_name"]

    if not lex_cols:
        return go.Figure()
    
    global_means = lex_df[lex_cols].astype(float).mean()
    top4 = global_means.nlargest(8).index.tolist()

    df2 = df.sort_values("release_year", na_position="last")
    lex_df = lex_df.reindex(df2["album_name"])
    
    labels = [a[:18] + "…" if len(str(a)) > 12 else str(a) for a in lex_df.index]

    fig = go.Figure()
    for i, (col, _) in enumerate(zip(top4, range(len(top4)))):
        color = LEXICAL_COLORS.get(col, FALLBACK[i % len(FALLBACK)])
        fig.add_trace(go.Scatter(
            x=lex_df.index,
            y=lex_df[col].astype(float),
            name=col.capitalize().replace("_", " "),
            mode="lines+markers",
            stackgroup="one",
            line=dict(color=color, width=1.5),
            fillcolor=color,
            hovertemplate=f"<b>%{{x}}</b><br>{col.capitalize()} : %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT, height=540,
        xaxis=dict(
                tickangle=-30,
                tickfont=dict(size=10),
                tickmode="array",
                tickvals=list(lex_df.index),
                ticktext=labels,
            ),
        yaxis=dict(gridcolor="#f0f0f0", tickformat=".3f"),
        legend=dict(orientation="h", y=-0.35, font=dict(size=10)),
    )
    return fig

def centroid_chart(names: list[str], embs: np.ndarray, selected: list[str] = None) -> go.Figure:
    selected = selected or []
    pca = PCA(n_components=3)
    coords = pca.fit_transform(embs)

    palette = ["#1a5c38", "#185fa5", "#a32d2d", "#534ab7", "#854f0b", "#0f6e56",
               "#c9687a", "#cf835c", "#3b9ca1", "#708238"]

    fig = go.Figure()

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

def multi_radar_artists(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    # 1. Calculer la valeur maximale pour chaque métrique à travers tous les artistes
    # Cela permet de normaliser chaque axe indépendamment pour que le max soit à 1.
    max_vals = {}
    for k in RADAR_KEYS:
        if k in df.columns:
            # On gère les cas où la colonne pourrait être vide ou à 0
            max_val = df[k].astype(float).max()
            max_vals[k] = max_val if max_val > 0 else 1.0

    for i, (_, row) in enumerate(df.iterrows()):
        cats = []
        vals = []
        real_vals = [] # Pour garder une trace des valeurs originales au survol
        
        # 2. Construire les valeurs normalisées pour l'artiste courant
        for k in RADAR_KEYS:
            if k in row.index:
                raw_val = safe_float(row.get(k, 0))
                norm_val = raw_val / max_vals[k] # Normalisation : Valeur / Max de la colonne
                
                cats.append(RADAR_DISPLAY.get(k, k))
                vals.append(norm_val)
                real_vals.append(raw_val)
                
        if not cats:
            continue
            
        # Boucler les listes pour fermer le polygone du radar
        cats = cats + [cats[0]]
        vals = vals + [vals[0]]
        real_vals = real_vals + [real_vals[0]]

        line_color, fill_color = PALETTE_RADAR_MULTI_ARTISTS[i % len(PALETTE_RADAR_MULTI_ARTISTS)]
        
        # 3. Ajout de la trace avec un hovertemplate personnalisé
        fig.add_trace(go.Scatterpolar(
            r=vals, 
            theta=cats, 
            name=row.get("artist_name", f"Artiste {i}"),
            fill="toself",
            line=dict(color=line_color, width=2),
            fillcolor=fill_color,
            customdata=real_vals,
            # Le hovertemplate montre la vraie valeur, pas juste la valeur normalisée (0 à 1)
            hovertemplate="<b>%{theta}</b><br>Valeur brute: %{customdata:.3f}<br>Score relatif: %{r:.2f}<extra></extra>"
        ))

    fig.update_layout(
        polar=dict(
            # On garde range=[0, 1] car nos valeurs sont maintenant des pourcentages du max
            radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=9), gridcolor="#eee"),
            angularaxis=dict(tickfont=dict(size=10, color="#555")),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
        **_LAYOUT, height=400,
    )
    return fig

def scatter_ttr_streams_multi(df: pd.DataFrame, artist_names: list[str], stream_col: str = "streams") -> go.Figure:

    df2 = df[["track_name", "artist_name", "ttr", stream_col]].dropna()
    df2 = df2[df2[stream_col] > 0]
    if df2.empty:
        return go.Figure()

    fig = go.Figure()

    for i, artist in enumerate(artist_names):
        adf = df2[df2["artist_name"] == artist].copy()
        if adf.empty:
            continue

        color = COLORS_STREAM_TTR_MULTI[i % len(COLORS_STREAM_TTR_MULTI)]
        fig.add_trace(go.Scatter(
            x=adf["ttr"],
            y=adf[stream_col],
            mode="markers",
            name=artist,
            marker=dict(
                size=9,
                color=color,
                opacity=0.55,                
                line=dict(width=0.5, color="white"),
            ),
            text=adf["track_name"],
            hovertemplate="<b>%{text}</b><br>TTR : %{x:.3f}<br>Streams : %{y:,.0f}<extra></extra>",
        ))

        if len(adf) >= 3:
            x_vals = adf["ttr"].values.reshape(-1, 1)
            y_log  = np.log10(adf[stream_col].values)
            model  = LinearRegression().fit(x_vals, y_log)
            x_range = np.linspace(adf["ttr"].min(), adf["ttr"].max(), 100)
            y_pred  = 10 ** model.predict(x_range.reshape(-1, 1))

            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_pred,
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        **_LAYOUT, height=450,
        xaxis=dict(
            title="TTR",
            gridcolor="#ebebeb",
            range=[0.1, 1.0],
        ),
        yaxis=dict(
            title="Streams",
            type="log",                        
            gridcolor="#ebebeb",
            tickvals=[1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
            ticktext=["1K", "10K", "100K", "1M", "10M", "100M"],
        ),
        legend=dict(
            orientation="h",
            y=1.08, x=0.5, xanchor="center",
            font=dict(size=11),
        ),
    )
    return fig