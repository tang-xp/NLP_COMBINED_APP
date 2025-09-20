import os
import warnings
from collections import Counter

import joblib
import numpy as np
import spacy
import sklearn_crfsuite
from scipy.stats import uniform
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics
from sklearn.model_selection import RandomizedSearchCV
from .xp_features import FeatureExtractor

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath(__file__))


def load_data():
    try:
        train_path = os.path.join(script_dir, "xp_train_data.pkl")
        test_path = os.path.join(script_dir, "xp_test_data.pkl")
        train_data = joblib.load(train_path)
        test_data = joblib.load(test_path)
        if not train_data or not test_data:
            raise ValueError("Empty dataset")
        print(f"Loaded {len(train_data)} train / {len(test_data)} test samples.")
        return train_data, test_data
    except Exception as e:
        raise RuntimeError(f"Data load failed: {e}")


train_data, test_data = load_data()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

feature_extractor = FeatureExtractor()


def sent2features(tokens):
    if not tokens:
        return []
    doc = nlp(" ".join(tokens))
    if len(doc) != len(tokens):
        return []
    chunk_info = [{"tag": "O", "is_event_chunk": False} for _ in doc]
    for chunk in doc.noun_chunks:
        has_event = any(t.text.lower() in feature_extractor.event_words for t in chunk)
        if len(chunk) == 1:
            chunk_info[chunk.start]["tag"] = "S-CHUNK"
        else:
            chunk_info[chunk.start]["tag"] = "B-CHUNK"
            for i in range(chunk.start + 1, chunk.end - 1):
                chunk_info[i]["tag"] = "I-CHUNK"
            chunk_info[chunk.end - 1]["tag"] = "E-CHUNK"
        if has_event:
            for i in range(chunk.start, chunk.end):
                chunk_info[i]["is_event_chunk"] = True
    return [feature_extractor.enhanced_word_features(doc, i, chunk_info)
            for i in range(len(doc))]


def prepare_features(data):
    X, y = [], []
    for i, (tokens, labels) in enumerate(data):
        feats = sent2features(tokens)
        if len(feats) != len(labels):
            continue
        X.append(feats)
        y.append(labels)
    return X, y


X_train, y_train = prepare_features(train_data)
X_test, y_test = prepare_features(test_data)
print(f"Prepared {len(X_train)} train / {len(X_test)} test sentences.")


def balance_data(X, y, max_ratio=3):
    entity_counts = Counter(l for labels in y for l in labels if l != "O")
    balanced_X, balanced_y = [], []
    for x, labels in zip(X, y):
        balanced_X.append(x)
        balanced_y.append(labels)
        ents = [l for l in labels if l != "O"]
        if ents and min(entity_counts[e] for e in set(ents)) < max(entity_counts.values()) / 10:
            for _ in range(min(max_ratio, 2)):
                balanced_X.append(x)
                balanced_y.append(labels)
    return balanced_X, balanced_y


X_train_bal, y_train_bal = balance_data(X_train, y_train)

crf = sklearn_crfsuite.CRF(algorithm="lbfgs", max_iterations=100, all_possible_transitions=True)
params = {"c1": uniform(0, 0.5), "c2": uniform(0, 0.3)}
f1_scorer = make_scorer(metrics.flat_f1_score, average="weighted")  # Fixed: use metrics.flat_f1_score

rs = RandomizedSearchCV(crf, params, n_iter=20, cv=3, scoring=f1_scorer,
                        verbose=1, n_jobs=-1, random_state=42)
rs.fit(X_train_bal, y_train_bal)
crf = rs.best_estimator_
print(f"Best params: {rs.best_params_}")
print(f"Best CV F1: {rs.best_score_:.4f}")

y_pred = crf.predict(X_test)
labels = sorted({l for seq in (y_train + y_test) for l in seq if l != "O"})
if labels:
    print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))
    print(f"Test F1: {metrics.flat_f1_score(y_test, y_pred, average='weighted'):.3f}")

joblib.dump(crf, os.path.join(script_dir, "xp_crf_model.joblib"))
print("Model saved.")