"""TheGuard model zoo.

Three families of detector, all exposing the same interface used by the app
(`.predict_proba(list_of_str)`):

  1. baseline()   - the original TF-IDF (word + char) + Logistic Regression.
  2. hybrid()     - TF-IDF  +  engineered scam-signal features (features.py).
  3. embeddings() - pretrained sentence-transformer embeddings + Logistic
                    Regression (a "frozen" transformer, no fine-tuning needed).

`ensemble()` averages the hybrid and embedding detectors.
"""
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from features import SignalFeatures

# --------------------------- shared text vectorisers ----------------------
def _word_vec():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)


def _char_vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           sublinear_tf=True, min_df=2)


def _clf(C=4.0):
    return LogisticRegression(max_iter=3000, C=C, class_weight="balanced")


# ------------------------------- baseline ---------------------------------
def baseline():
    """Original model: word + char TF-IDF fused, then logistic regression."""
    feats = FeatureUnion([("word", _word_vec()), ("char", _char_vec())])
    return Pipeline([("feats", feats), ("clf", _clf())])


# -------------------------------- hybrid ----------------------------------
class _ScaledSignals(BaseEstimator, TransformerMixin):
    """Engineered signals, scaled to a comparable range for the classifier."""

    def fit(self, X, y=None):
        self.sf_ = SignalFeatures().fit(X)
        self.scaler_ = MaxAbsScaler().fit(self.sf_.transform(X))
        return self

    def transform(self, X):
        return self.scaler_.transform(self.sf_.transform(X))


def hybrid(C=4.0):
    """TF-IDF (wording) + engineered signals (reasoning) + logistic regression."""
    feats = FeatureUnion([
        ("word", _word_vec()),
        ("char", _char_vec()),
        ("signals", _ScaledSignals()),
    ])
    return Pipeline([("feats", feats), ("clf", _clf(C))])


# ------------------------- transformer embeddings -------------------------
DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceFeaturizer(BaseEstimator, TransformerMixin):
    """Wraps a pretrained sentence-transformer as an sklearn transformer.

    Only the model *name* is kept on the instance, so pickling a fitted
    pipeline stays lightweight and version-safe; the torch model is loaded
    lazily on first use (and cached under ~/.cache/huggingface)."""

    def __init__(self, model_name=DEFAULT_ST_MODEL, batch_size=64):
        self.model_name = model_name
        self.batch_size = batch_size

    def _model(self):
        if not hasattr(self, "model_"):
            from sentence_transformers import SentenceTransformer
            self.model_ = SentenceTransformer(self.model_name)
        return self.model_

    def fit(self, X, y=None):
        self._model()  # ensure it is loadable/downloaded at fit time
        return self

    def transform(self, X):
        if isinstance(X, str):
            X = [X]
        emb = self._model().encode(list(X), batch_size=self.batch_size,
                                   normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float64)

    # keep pickles small & robust: drop the heavy torch object on (de)serialise
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("model_", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def embeddings(model_name=DEFAULT_ST_MODEL, C=8.0):
    """Pretrained sentence embeddings + logistic regression."""
    return Pipeline([
        ("emb", SentenceFeaturizer(model_name)),
        ("clf", LogisticRegression(max_iter=3000, C=C, class_weight="balanced")),
    ])


# -------------------------------- ensemble --------------------------------
class EnsembleModel(BaseEstimator):
    """Average the fraud probability of several detectors."""

    def __init__(self, models=None, weights=None):
        self.models = models or []
        self.weights = weights

    def fit(self, X, y=None):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        if isinstance(X, str):
            X = [X]
        ps = np.vstack([np.asarray(m.predict_proba(X))[:, 1] for m in self.models])
        w = np.ones(len(ps)) if self.weights is None else np.asarray(self.weights, dtype=float)
        w = w / w.sum()
        p = (ps * w[:, None]).sum(axis=0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def ensemble(models, weights=None):
    return EnsembleModel(models=list(models), weights=weights)
