from pathlib import Path

DB_PATH = Path("data/warehouse.duckdb")

GREEN_DARK   = "#1a5c38"
GREEN_MID    = "#2e8a57"
GREEN_LIGHT  = "#5dbf8a"
GREEN_BG     = "#e8f5ee"
GREEN_CARD   = "#0f3d25"

COLORS = {
    "positive":  "#2e8a57",
    "neutral":   "#88878061",
    "negative":  "rgba(163,45,45,0.7)",
    "joie":      "#1a5c38",
    "tristesse": "#185fa5",
    "colere":    "#a32d2d",
    "peur":      "#534ab7",
    "surprise":  "#854f0b",
    "degout":    "#0f6e56",
    "primary":   GREEN_DARK,
    "primary_light": GREEN_LIGHT,
}

EMOTION_LABELS = ["joie", "tristesse", "colere", "peur", "surprise", "degout"]
EMOTION_DISPLAY = {
    "joie":      "Joie",
    "tristesse": "Tristesse",
    "colere":    "Colère",
    "peur":      "Peur",
    "surprise":  "Surprise",
    "degout":    "Dégoût",
}

SENTIMENT_COLS = {
    "sentiment_positive": "Positif",
    "sentiment_neutral":  "Neutre",
    "sentiment_negative": "Négatif",
}

NLP_FEATURES = [
    "avg_ttr", "avg_rhyme_density", "avg_semantic_density",
    "avg_pos_verb_ratio", "avg_pos_noun_ratio", "avg_pos_adj_ratio",
    "avg_pronoun_i_ratio", "avg_lexical_diversity", "avg_hapax_ratio",
]
NLP_FEATURES_DISPLAY = {
    "avg_ttr":              "TTR",
    "avg_rhyme_density":    "Densité rimes",
    "avg_semantic_density": "Densité sémantique",
    "avg_pos_verb_ratio":   "Ratio verbes",
    "avg_pos_noun_ratio":   "Ratio noms",
    "avg_pos_adj_ratio":    "Ratio adjectifs",
    "avg_pronoun_i_ratio":  "Ratio je/j'",
    "avg_lexical_diversity":"Diversité lexicale",
    "avg_hapax_ratio":      "Ratio hapax",
}

AUDIO_FEATURES = [
    "tempo", "beat_strength", "rms_mean", "dynamic_range",
    "spectral_centroid", "brightness", "warmth", "roughness",
]
AUDIO_FEATURES_DISPLAY = {
    "tempo":             "Tempo",
    "beat_strength":     "Force du beat",
    "rms_mean":          "Volume moyen",
    "dynamic_range":     "Dynamique",
    "spectral_centroid": "Centroïde spectral",
    "brightness":        "Brillance",
    "warmth":            "Chaleur",
    "roughness":         "Rugosité",
}

RADAR_KEYS = [
    "avg_rhyme_density",
    "avg_ttr",
    "avg_semantic_density",
    "avg_pronoun_i_ratio",
    "avg_pos_verb_ratio",
    "avg_hapax_ratio",
]
RADAR_DISPLAY = {
    "avg_rhyme_density":    "Rimes",
    "avg_ttr":              "TTR",
    "avg_semantic_density": "Sémantique",
    "avg_pronoun_i_ratio":  "Auto-réf.",
    "avg_pos_verb_ratio":   "Verbes",
    "avg_hapax_ratio":      "Hapax",
}

LEXICAL_FIELD_DISPLAY = {
    "avg_lexical_violence": "Violence",
    "avg_lexical_money":    "Argent",
    "avg_lexical_street":   "Street",
    "avg_lexical_love":     "Amour",
}

ARTIST_DISPLAY_NAMES = {
    "Ntmojo": "Suprême NTM",
    # autres corrections...
}

LEXICAL_COLORS = {
        "argent":       "rgba(133,79,11,0.35)",
        "rue":          "rgba(26,92,56,0.35)",
        "famille":      "rgba(24,95,165,0.35)",
        "drogue":       "rgba(83,74,183,0.35)",
        "célébrité":    "rgba(136,135,128,0.35)",
        "spiritualité": "rgba(15,110,86,0.35)",
        "amour_perdu":  "rgba(163,45,45,0.35)",
        "violence":     "rgba(163,45,45,0.35)",
        "succès":       "rgba(46,138,87,0.35)",
        "échec":        "rgba(24,95,165,0.35)",
        "liberté":      "rgba(93,191,138,0.35)",
        "prison":       "rgba(83,74,183,0.35)",
        "mort":         "rgba(26,92,56,0.35)",
        "fête":         "rgba(133,79,11,0.35)",
        "sport":        "rgba(15,110,86,0.35)",
        "mode":         "rgba(136,135,128,0.35)",
        "voitures":     "rgba(133,79,11,0.35)",
    }
FALLBACK = [
        "rgba(244,162,97,0.35)",
        "rgba(231,111,81,0.35)",
        "rgba(38,70,83,0.35)",
        "rgba(42,157,143,0.35)",
        "rgba(233,196,106,0.35)",
        "rgba(168,218,220,0.35)",
    ]

ARTIST_DISPLAY_NAMES_INV = {v: k for k, v in ARTIST_DISPLAY_NAMES.items()}


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

EMOTION_COLORS_HEATMAP = {
        "joie":        "#2e8a57",
        "amour":       "#c9687a",
        "sympathie":   "#cf835c",
        "tristesse":   "#185fa5",
        "colère":      "#a32d2d",
        "peur":        "#534ab7",
        "surprise":    "#854f0b",
        "dégoût":      "#0f6e56",
        "nostalgie":   "#888780",
        "honte":       "#7d5168",
        "embarras":    "#da9fa6",
        "culpabilité": "#4f5d75",
        "envie":       "#708238",
        "jalousie":    "#3a4a18",
        "gratitude":   "#c29946",
        "indignation": "#bd3a3a",
        "mépris":      "#5a6b7c",
        "espoir":      "#3b9ca1",
        "désespoir":   "#1c2836",
        "méfiance":    "#6b725c",
    }
FALLBACK_EMOTION_HEATMAP = ["#f4a261", "#e76f51", "#264653", "#2a9d8f", "#e9c46a", "#a8dadc", "#457b9d", "#e63946"]

EMOTION_COLORS_RGBA = {
        "joie":        "rgba(46,138,87,0.6)",
        "amour":       "rgba(201,104,122,0.6)",
        "sympathie":   "rgba(207,131,92,0.6)",
        "tristesse":   "rgba(24,95,165,0.6)",
        "colère":      "rgba(163,45,45,0.6)",
        "peur":        "rgba(83,74,183,0.6)",
        "surprise":    "rgba(133,79,11,0.6)",
        "dégoût":      "rgba(15,110,86,0.6)",
        "nostalgie":   "rgba(136,135,128,0.6)",
        "honte":       "rgba(125,81,104,0.6)",
        "embarras":    "rgba(218,159,166,0.6)",
        "culpabilité": "rgba(79,93,117,0.6)",
        "envie":       "rgba(112,130,56,0.6)",
        "jalousie":    "rgba(58,74,24,0.6)",
        "gratitude":   "rgba(194,153,70,0.6)",
        "indignation": "rgba(189,58,58,0.6)",
        "mépris":      "rgba(90,107,124,0.6)",
        "espoir":      "rgba(59,156,161,0.6)",
        "désespoir":   "rgba(28,40,54,0.6)",
        "méfiance":    "rgba(107,114,92,0.6)",
    }
FALLBACK_EMOTION_RGBA = [
        "rgba(244,162,97,0.6)",
        "rgba(231,111,81,0.6)",
        "rgba(38,70,83,0.6)",
        "rgba(42,157,143,0.6)",
        "rgba(233,196,106,0.6)",
        "rgba(168,218,220,0.6)",
        "rgba(69,123,157,0.6)",
        "rgba(230,57,70,0.6)",
    ]

PALETTE_RADAR_MULTI_ARTISTS = [
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

COLORS_STREAM_TTR_MULTI = ["#2e8a57","#c9687a","#cf835c","#185fa5","#a32d2d","#534ab7","#854f0b","#0f6e56","#852F7F","#7d5168","#da9fa6","#708238","#c29946","#bd3a3a","#3b9ca1","#6b725c"]