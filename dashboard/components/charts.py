from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import COLORS, EMOTION_LABELS, EMOTION_DISPLAY

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


# ── Radar ───────────────────────────────────────────────────────────────────

def radar_chart(values: dict, title: str = "", compare: dict | None = None) -> go.Figure:
    cats = list(values.keys()) + [list(values.keys())[0]]
    vals = list(values.values()) + [list(values.values())[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats, fill="toself", name="Artiste",
        line=dict(color=COLORS["primary"], width=2),
        fillcolor="rgba(26,92,56,0.18)",
    ))
    if compare:
        c_cats = list(compare.keys()) + [list(compare.keys())[0]]
        c_vals = list(compare.values()) + [list(compare.values())[0]]
        fig.add_trace(go.Scatterpolar(
            r=c_vals, theta=c_cats, fill="toself", name="Corpus moyen",
            line=dict(color="#888780", width=1.5, dash="dot"),
            fillcolor="rgba(136,135,128,0.12)",
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9), gridcolor="#eee"),
            angularaxis=dict(tickfont=dict(size=10, color="#555")),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=bool(compare),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        **_LAYOUT, height=280,
    )
    return fig


# ── Sentiment line ───────────────────────────────────────────────────────────

def sentiment_line(df: pd.DataFrame, x_col: str = "album_name") -> go.Figure:
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
    cols_present = [c for c in EMOTION_LABELS if c in df.columns]
    if not cols_present:
        return go.Figure()
    z = df[cols_present].values.T
    fig = go.Figure(go.Heatmap(
        z=z,
        x=df.index.tolist(),
        y=[EMOTION_DISPLAY.get(c, c) for c in cols_present],
        colorscale=[[0, "#f0f7f3"], [0.5, "#5dbf8a"], [1, "#0f3d25"]],
        text=np.round(z, 2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=9),
        showscale=True,
        colorbar=dict(thickness=10, tickfont=dict(size=9)),
    ))
    fig.update_layout(
        **_LAYOUT,
        height=max(220, 40 * len(cols_present) + 60),
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


# ── Lexical fields bars ──────────────────────────────────────────────────────

def lexical_bars(values: dict[str, float]) -> go.Figure:
    labels = list(values.keys())
    vals   = list(values.values())
    colors = [COLORS["primary"] if v == max(vals) else COLORS["primary_light"] for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        **_LAYOUT, height=180,
        xaxis=dict(range=[0, max(vals) * 1.3 if vals else 1], showgrid=False, visible=False),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
    )
    return fig


# ── Top words bars ───────────────────────────────────────────────────────────

def top_words_bar(words: list[str], counts: list[int] | None = None) -> go.Figure:
    words = words[:20][::-1]
    y     = counts[:20][::-1] if counts else list(range(len(words), 0, -1))
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
        yaxis=dict(tickfont=dict(size=10)),
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
        **_LAYOUT, height=300,
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
        height=200,
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
    fig = go.Figure(go.Bar(
        x=sub[metric], y=sub["artist_name"], orientation="h",
        marker_color=colors,
        text=sub[metric].round(3),
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        **_LAYOUT, height=max(220, 40 * len(sub) + 80),
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
    df2 = df.sort_values("release_year", na_position="last")
    x   = df2["album_name"]

    for col, label, color in pairs:
        if col not in df2.columns:
            continue
        yax = "y2" if col == "album_ttr" else "y"
        fig.add_trace(go.Scatter(
            x=x, y=df2[col], name=label,
            mode="lines+markers", yaxis=yax,
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
        ))

    fig.update_layout(
        **_LAYOUT, height=260,
        xaxis=dict(tickangle=-30, gridcolor="#f0f0f0", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#f0f0f0", title="Vocabulaire", tickfont=dict(size=10)),
        yaxis2=dict(overlaying="y", side="right", title="TTR",
                    tickformat=".3f", tickfont=dict(size=10)),
        legend=dict(orientation="h", y=-0.3, font=dict(size=10)),
    )
    return fig
