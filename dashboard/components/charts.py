from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
from data.transforms import safe_float

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

RADAR_AUDIO_KEYS = {
    "tempo":          "Rapidité",
    "beat_strength":  "Puissance",
    "brightness":     "Brillance",
    "warmth":         "Chaleur",
    "roughness":      "Rugosité",
    "onset_rate":     "Flow",
}

# Plages réelles pour normalisation absolue (évite l'écrasement sur un seul artiste)
RADAR_AUDIO_RANGES = {
    "tempo":         (60,   199),
    "beat_strength": (1.7,  16.4),
    "brightness":    (0.007, 0.552),
    "warmth":        (0.137, 0.709),
    "roughness":     (0.108, 2.087),
    "onset_rate":    (0.033, 7.07),
}

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
    cols_present = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not cols_present or df.empty:
        return go.Figure()

    # Garde seulement les 9 émotions avec moyenne la plus haute
    means = df[cols_present].mean()
    cols_present = means.sort_values(ascending=False).head(9).index.tolist()

    # Tronque les noms d'albums
    albums = [a[:15] + "…" if len(str(a)) > 15 else str(a) for a in df.index.tolist()]
    emotions = [EMOTION_DISPLAY.get(c, c.capitalize()) for c in cols_present]
    n = len(cols_present)

    def interpolate_green(i, total):
        t = i / max(total - 1, 1)
        r = int(166 + (26  - 166) * t)
        g = int(210 + (92  - 210) * t)
        b = int(140 + (56  - 140) * t)
        return f"rgb({r},{g},{b})"

    fig = go.Figure()

    for i, (col, emo_label) in enumerate(zip(cols_present, emotions)):
        vals = df[col].fillna(0).values
        vmax = vals.max() if vals.max() > 0 else 1
        # Écart amplifié : 4px → 44px
        sizes = [3 + (v / vmax) ** 0.6 * 19 for v in vals]
        color = interpolate_green(i, n)

        fig.add_trace(go.Scatter(
            x=albums,
            y=[emo_label] * len(albums),
            mode="markers",
            name=emo_label,
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.9,
                line=dict(color="#ffffff", width=1.5),
            ),
            text=[f"<b>{emo_label}</b><br>{v:.3f}" for v in vals],
            hovertemplate="%{x}<br>%{text}<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT,
        height=max(300, 55 * len(cols_present) + 80),  # 44 → 55
        showlegend=False,
        xaxis=dict(
            tickangle=-35,
            tickfont=dict(size=13, color="#888"),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=15, color="#555"),
            showgrid=True,
            gridcolor="#f0f0f0",
            gridwidth=1,
            zeroline=False,
        ),
    )
    return fig

# ── Lexical fields bars ──────────────────────────────────────────────────────

def lexical_bars(values: dict[str, float]) -> go.Figure:
    labels = list(values.keys())
    vals   = list(values.values())
    total  = sum(vals) if sum(vals) > 0 else 1
    pcts   = [v / total * 100 for v in vals]
    colors = [COLORS["primary"] if v == max(vals) else COLORS["primary_light"] for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        **_LAYOUT,
        height=max(300, 32 * len(labels) + 60),
        xaxis=dict(range=[0, max(vals) * 1.35 if vals else 1], showgrid=False, visible=False),
        yaxis=dict(tickfont=dict(size=13), autorange="reversed"),
        bargap=0.2,
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
        text=texts,
        textposition="outside",
        textfont=dict(size=11, color="#555", family="DM Sans"),
        showlegend=False,
        hovertemplate="%{y} : %{text}<extra></extra>",
        width=0.5,
    ))

    # Ligne verticale moyenne corpus
    for r in rows:
        fig.add_shape(
            type="line",
            x0=r["avg"], x1=r["avg"],
            y0=labels.index(r["label"]) - 0.35,
            y1=labels.index(r["label"]) + 0.35,
            line=dict(color="#888780", width=2, dash="dot"),
        )

    # Annotation légende manuelle
    fig.add_annotation(
        x=0.98, y=-0.08,
        xref="paper", yref="paper",
        text="<b>——</b> Artiste   <b style='color:#888'>····</b> Moyenne corpus",
        showarrow=False,
        font=dict(size=10, color="#888", family="DM Sans"),
        align="right",
    )

    fig.update_layout(
        **_LAYOUT,
        barmode="overlay",
        height=max(300, 52 * len(rows) + 80),
        bargap=0.3,
        xaxis=dict(visible=False, range=[0, 1.45]),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
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

def emotion_donut_chart(avg_emotion_scores: str | None) -> go.Figure:
    if not avg_emotion_scores:
        return go.Figure()

    scores = json.loads(avg_emotion_scores)
    scores = {k: v for k, v in scores.items() if v > 0.01}
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    n = len(scores)
    labels = [EMOTION_DISPLAY.get(k, k.capitalize()) for k in scores.keys()]
    values = list(scores.values())
    dominant = labels[0] if labels else ""

    # Dégradé vert clair → vert foncé selon le rang
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