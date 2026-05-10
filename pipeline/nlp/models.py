"""
models.py
=========
Chargement lazy des modèles ML (spaCy, SentenceTransformer)
et wrapper LDA sklearn.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from pipeline.nlp.helpers import custom_tokenizer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Instances lazy (initialisées au premier appel)
# ─────────────────────────────────────────────────────────────────────────────

_nlp = None
_sbert_inst = None


def get_nlp():
    """Charge fr_core_news_sm (ou en_core_web_sm en fallback)."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("fr_core_news_sm")
        except OSError:
            logger.warning("fr_core_news_sm absent — fallback en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def get_sbert():
    """Charge le modèle SentenceTransformer multilingue."""
    global _sbert_inst
    if _sbert_inst is None:
        from sentence_transformers import SentenceTransformer
        _sbert_inst = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _sbert_inst


# ─────────────────────────────────────────────────────────────────────────────
# LDA
# ─────────────────────────────────────────────────────────────────────────────

class LDAModel:
    """
    Wrapper sklearn LDA.
    Expose la même interface que les appels gensim utilisés ailleurs dans
    le pipeline : transform() → [(topic_id, prob), …], top_words().
    """

    def __init__(self, num_topics: int = 12):
        self.num_topics = num_topics
        self.vectorizer = CountVectorizer(
            tokenizer=custom_tokenizer,
            min_df=3,
            max_df=0.70,
            lowercase=True,
        )
        self.lda = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=50,
            learning_method="batch",
            random_state=42,
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
        )
        self.vocab: list[str] = []

    def fit(self, corpus_texts: list[str]) -> "LDAModel":
        dtm = self.vectorizer.fit_transform(corpus_texts)
        self.lda.fit(dtm)
        self.vocab = self.vectorizer.get_feature_names_out().tolist()
        return self

    def transform(self, text: str) -> list[tuple[int, float]]:
        vec = self.vectorizer.transform([text])
        probs = self.lda.transform(vec)[0]
        return [(i, float(p)) for i, p in enumerate(probs)]

    def top_words(self, topic_id: int, topn: int = 5) -> list[str]:
        comp = self.lda.components_[topic_id]
        indices = comp.argsort()[-topn:][::-1]
        return [self.vocab[i] for i in indices]


def build_lda(corpus_texts: list[str], num_topics: int = 8) -> LDAModel:
    """Entraîne et retourne un LDAModel sur le corpus complet."""
    model = LDAModel(num_topics=num_topics)
    model.fit(corpus_texts)
    logger.info(
        "LDA entraîné — %d topics | vocabulaire %d mots",
        num_topics, len(model.vocab),
    )
    return model
