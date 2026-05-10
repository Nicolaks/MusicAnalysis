"""
analyzers/track.py
==================
Analyse NLP d'une seule chanson (niveau 1).
Retourne un dict prêt à être inséré dans tracks_analysis.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Optional

import textstat
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.nlp.config import ALL_STOPWORDS
from pipeline.nlp.helpers import (
    avg, clean_lyrics, count_syllables, custom_tokenizer,
    filtered_tokens, safe_div, yule_k, score_emotions, score_lexical_fields, dominant_emotions
)
from pipeline.nlp.models import get_nlp, get_sbert


def analyze_track(
    track_id: int,
    artist_id: int,
    album_id: Optional[int],
    track_name: str,
    artist_name: str,
    album_name: Optional[str],
    raw_lyrics: str,
    isrc: Optional[str] = None,
    lda_model=None,
) -> dict:
    """
    Calcule toutes les métriques NLP d'une chanson et retourne un dict
    correspondant aux colonnes de tracks_analysis.
    """
    lyrics = clean_lyrics(raw_lyrics)
    r: dict = {
        "track_id": track_id, "artist_id": artist_id, "album_id": album_id,
        "track_name": track_name, "artist_name": artist_name, "album_name": album_name,
        "isrc": isrc,
    }

    # ── 2. Stats lexicales ────────────────────────────────────────────────────
    alpha = filtered_tokens(lyrics)
    sents = sent_tokenize(lyrics, language="french")
    r["word_count"]          = len(alpha)
    r["unique_word_count"]   = len(set(alpha))
    r["ttr"]                 = safe_div(len(set(alpha)), len(alpha))
    r["sentence_count"]      = len(sents)
    r["avg_sentence_length"] = safe_div(len(alpha), len(sents))
    freq = FreqDist(alpha)
    r["top10_words"] = json.dumps([w for w, _ in freq.most_common(10)], ensure_ascii=False)

    # ── 3. Sémantique ─────────────────────────────────────────────────────────
    try:
        tfidf = TfidfVectorizer(
            max_features=10,
            stop_words=list(ALL_STOPWORDS),
            tokenizer=custom_tokenizer,
        )
        tfidf.fit_transform([lyrics])
        r["tfidf_top_keywords"] = json.dumps(
            tfidf.get_feature_names_out().tolist(), ensure_ascii=False
        )
    except Exception:
        r["tfidf_top_keywords"] = None

    if lda_model:
        try:
            topics = lda_model.transform(lyrics)
            best   = max(topics, key=lambda x: x[1])
            r["lda_dominant_topic"] = int(best[0])
            r["lda_topic_keywords"] = json.dumps(
                lda_model.top_words(best[0], topn=5), ensure_ascii=False
            )
        except Exception:
            r["lda_dominant_topic"] = r["lda_topic_keywords"] = None
    else:
        r["lda_dominant_topic"] = r["lda_topic_keywords"] = None

    try:
        emb = get_sbert().encode(lyrics[:1000])
        r["embedding_norm"] = float(np.linalg.norm(emb))
    except Exception:
        r["embedding_norm"] = None

    # ── 4. Stylométrie spaCy ──────────────────────────────────────────────────
    doc   = get_nlp()(lyrics[:100_000])
    total = sum(1 for t in doc if not t.is_space)
    pos_c: dict[str, int] = {}
    for t in doc:
        if not t.is_space:
            pos_c[t.pos_] = pos_c.get(t.pos_, 0) + 1

    for pos, key in [
        ("NOUN", "noun"), ("VERB", "verb"), ("ADJ", "adj"),
        ("ADV", "adv"), ("PRON", "pron"),
    ]:
        r[f"pos_{key}_ratio"] = safe_div(pos_c.get(pos, 0), total)

    r["pronoun_i_ratio"]   = safe_div(
        sum(1 for t in doc if t.lower_ in {"je", "j'"}), total
    )
    r["pronoun_we_ratio"]  = safe_div(
        sum(1 for t in doc if t.lower_ in {"on", "nous"}), total
    )
    r["pronoun_you_ratio"] = safe_div(
        sum(1 for t in doc if t.lower_ in {"tu", "vous", "t'"}), total
    )

    # ── 5. Rimes & structure ──────────────────────────────────────────────────
    lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
    rhyme_count  = 0
    syl_per_line = []

    for line in lines:
        ws = line.split()
        if ws:
            last = re.sub(r"[^\w]", "", ws[-1]).lower()
            syl_per_line.append(sum(count_syllables(w) for w in ws))
            if len(last) >= 3:
                rhyme_count += sum(
                    1 for other in lines
                    if other != line and other.split()
                    and re.sub(r"[^\w]", "", other.split()[-1]).lower()[-3:] == last[-3:]
                )

    r["rhyme_density"]      = safe_div(rhyme_count, len(lines))
    r["avg_syllables_line"] = avg(syl_per_line)
    line_counts = Counter(lines)
    repeated = sum(v for v in line_counts.values() if v >= 3)
    r["repetition_ratio"] = safe_div(repeated, len(lines))
    r["chorus_detected"]  = repeated > 0

    # ── 6. Lisibilité ─────────────────────────────────────────────────────────
    r["flesch_reading_ease"]  = textstat.flesch_reading_ease(lyrics)
    r["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(lyrics)
    r["smog_index"]           = textstat.smog_index(lyrics)
    r["avg_word_length"]      = safe_div(sum(len(t) for t in alpha), len(alpha))

    # ── 7. Analyse avancée ────────────────────────────────────────────────────
    content = [t for t in doc if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"} and not t.is_space]
    r["semantic_density"]  = safe_div(len(content), total)
    r["lexical_diversity"] = yule_k(alpha)
    fd2   = FreqDist(alpha)
    hapax = sum(1 for _, c in fd2.items() if c == 1)
    r["hapax_ratio"] = safe_div(hapax, len(set(alpha)))
    
    # ── 8. Émotions & champs lexicaux ────────────────────────────────────────────
    emotion_scores = score_emotions(alpha)
    lex_scores     = score_lexical_fields(alpha)

    r["emotion_scores"]          = json.dumps(emotion_scores, ensure_ascii=False)
    r["dominant_emotions"]       = json.dumps(dominant_emotions(emotion_scores), ensure_ascii=False)
    r["lexical_field_scores"]    = json.dumps(lex_scores, ensure_ascii=False)
    r["dominant_lexical_fields"] = json.dumps(dominant_emotions(lex_scores), ensure_ascii=False)

    return r
