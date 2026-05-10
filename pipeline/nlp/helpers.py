"""
helpers.py
==========
Fonctions utilitaires pures : mathématiques, nettoyage de texte, tokenisation,
comptage de syllabes. Aucune dépendance aux modèles ML ou à la BDD.
"""

from __future__ import annotations

import re

import numpy as np
import pyphen
from nltk.tokenize import word_tokenize

from pipeline.nlp.config import *

# ─────────────────────────────────────────────────────────────────────────────
# Texte
# ─────────────────────────────────────────────────────────────────────────────

def clean_lyrics(text: str) -> str:
    """Supprime les balises [Couplet], les sauts de ligne excessifs, etc."""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def filtered_tokens(text: str | list[str]) -> list[str]:
    """
    Tokenise et filtre : garde uniquement les tokens alphabétiques,
    hors stopwords, de longueur > 2.
    """
    if isinstance(text, list):
        tokens = [t.lower() for t in text]
    else:
        tokens = word_tokenize(text.lower(), language="french")

    return [
        t for t in tokens
        if t.isalpha() and t not in ALL_STOPWORDS and len(t) > 2
    ]


def custom_tokenizer(text: str) -> list[str]:
    """Tokenizer compatible sklearn (même logique que filtered_tokens)."""
    return filtered_tokens(text)


# ─────────────────────────────────────────────────────────────────────────────
# Syllabes
# ─────────────────────────────────────────────────────────────────────────────

_dic = pyphen.Pyphen(lang="fr_FR")

def count_syllables(word: str) -> int:
    """Compte les syllabes d'un mot français via pyphen, avec fallback voyelles."""
    h = _dic.inserted(word)
    if "-" in h:
        return h.count("-") + 1
    return max(1, len(re.findall(r"[aeiouàâäéèêëîïôöùûü]", word.lower())))


# ─────────────────────────────────────────────────────────────────────────────
# Mathématiques / statistiques
# ─────────────────────────────────────────────────────────────────────────────

def safe_div(a, b) -> float:
    return float(a) / float(b) if b else 0.0


def avg(lst: list, default: float = 0.0) -> float:
    return float(np.mean(lst)) if lst else default


def std(lst: list, default: float = 0.0) -> float:
    return float(np.std(lst)) if lst else default


def yule_k(tokens: list[str]) -> float:
    """Mesure de diversité lexicale de Yule-K (plus élevé = moins de diversité)."""
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    m1 = len(tokens)
    m2 = sum(v ** 2 for v in freq.values())
    return safe_div(10_000 * (m2 - m1), m1 ** 2) if m1 else 0.0


def shannon_entropy(values: list[float]) -> float:
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * np.log2(p) for p in probs)


def linear_trend(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    xs_a = np.array(xs, dtype=float)
    ys_a = np.array(ys, dtype=float)
    xs_c = xs_a - xs_a.mean()
    denom = (xs_c ** 2).sum()
    return float((xs_c * ys_a).sum() / denom) if denom else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Emotions / champs lexical
# ─────────────────────────────────────────────────────────────────────────────

def score_emotions(tokens: list[str]) -> dict[str, float]:
    """
    Retourne un score normalisé [0-1] par émotion selon la présence
    des mots du lexique dans les tokens filtrés.
    """
    if not tokens:
        return {e: 0.0 for e in EMOTION_LEXICON}
    token_set = set(tokens)
    return {
        emotion: safe_div(len(token_set & words), len(tokens))
        for emotion, words in EMOTION_LEXICON.items()
    }

def score_lexical_fields(tokens: list[str]) -> dict[str, float]:
    """
    Retourne un score normalisé [0-1] par champ lexical.
    """
    if not tokens:
        return {f: 0.0 for f in LEXICAL_FIELDS}
    token_set = set(tokens)
    return {
        field: safe_div(len(token_set & words), len(tokens))
        for field, words in LEXICAL_FIELDS.items()
    }

def dominant_emotions(scores: dict[str, float], top_n: int = 3) -> list[str]:
    """Retourne les N émotions dominantes (score > 0)."""
    return [
        e for e, s in sorted(scores.items(), key=lambda x: -x[1])
        if s > 0
    ][:top_n]
