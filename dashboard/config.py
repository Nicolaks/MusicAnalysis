from pathlib import Path

DB_PATH = Path("data/warehouse.duckdb")

GREEN_DARK   = "#1a5c38"
GREEN_MID    = "#2e8a57"
GREEN_LIGHT  = "#5dbf8a"
GREEN_BG     = "#e8f5ee"
GREEN_CARD   = "#0f3d25"

COLORS = {
    "positive":  "#2e8a57",
    "neutral":   "#888780",
    "negative":  "#a32d2d",
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
