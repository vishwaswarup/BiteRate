import json
import os
import pickle
import re

import numpy as np
from flask import Flask, render_template, request, jsonify

# Importing preprocessing also lets the pickled pipeline resolve clean_text.
from preprocessing import EMOJI_PATTERN, NEGATIONS, stop_words

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_REVIEW_CHARS = 5000
MAX_BATCH = 100
TOP_TERMS = 6
MAX_EXPLAINED_WORDS = 100  # caps the cost of explaining very long reviews

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
pipeline = None
try:
    with open(os.path.join(HERE, "model.pkl"), "rb") as f:
        pipeline = pickle.load(f)
    _classes = list(pipeline.classes_)
    _POS, _NEG = _classes.index("Positive"), _classes.index("Negative")
    pipeline.decision_function(["warm up"])  # first call is slow; pay it at startup, not on a request
    print("Model loaded.")
except Exception as e:  # the API reports 503 instead of faking predictions
    print(f"Could not load model.pkl ({e}).")

try:
    with open(os.path.join(HERE, "metrics.json")) as f:
        MODEL_INFO = json.load(f)
except Exception:
    MODEL_INFO = None

SAMPLE_REVIEWS = [
    "Absolutely loved the truffle pasta - rich, creamy, and cooked to perfection. Will definitely come back!",
    "Waited over an hour for cold, soggy fries and a burnt burger. Never again.",
    "The ambiance was nice but the food was just okay. Nothing special for the price they charge.",
    "Best sushi I've ever had outside of Japan. The chef clearly knows his craft.",
    "Terrible service. The waiter was rude and forgot half our order. Food was mediocre at best.",
    "Decent brunch spot. Pancakes were fluffy but the coffee was weak. Might try again.",
    "The wood-fired pizza was phenomenal - crispy crust, fresh mozzarella, perfect basil. A hidden gem!",
    "Overpriced and underwhelming. Paid $40 for a tiny portion of bland chicken. Very disappointed.",
    "Great cocktail menu and the appetizers were creative and delicious. Perfect for a date night!",
    "Found a hair in my soup. When I complained, the manager shrugged it off. Disgusting.",
    "Average experience overall. The pasta was slightly overcooked but dessert was surprisingly good.",
    "Hands down the best tacos in the city. The salsa verde is addictive. Five stars!",
]
SAMPLE_REVIEWS = [{"id": i + 1, "text": t} for i, t in enumerate(SAMPLE_REVIEWS)]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?|" + EMOJI_PATTERN.pattern.rstrip("+"))


def _is_negation(tok):
    return tok in NEGATIONS or tok.endswith("n't") or tok == "cannot"


def _utf16(text, i):
    """Python indexes code points; the browser indexes UTF-16 units (emojis count twice)."""
    return len(text[:i].encode("utf-16-le")) // 2


def _margin(texts):
    """How strongly the model leans positive (vs negative) for each text."""
    scores = pipeline.decision_function(texts)
    return scores[:, _POS] - scores[:, _NEG]


def _explain(text):
    """Key terms, and spans of the review that pushed towards positive or negative.

    Each distinct word is removed in turn and the review re-scored; the change is that
    word's effect. This accounts for every feature the model uses (words, negation
    marking and character n-grams), so it matches what the model actually did.
    """
    occurrences = {}
    for m in _TOKEN_RE.finditer(text):
        key = m.group().lower().replace("\u2019", "'")
        if key in stop_words and not _is_negation(key):
            continue
        occurrences.setdefault(key, []).append(m.span())
    words = list(occurrences)[:MAX_EXPLAINED_WORDS]
    if not words:
        return [], []

    variants = []
    for w in words:
        parts, last = [], 0
        for s, e in occurrences[w]:
            parts.append(text[last:s])
            last = e
        parts.append(text[last:])
        variants.append(" ".join(parts))
    effects = _margin([text])[0] - _margin(variants)

    ranked = sorted(zip(words, effects), key=lambda we: -abs(we[1]))
    terms = [{"term": w, "weight": round(float(e), 3)} for w, e in ranked[:TOP_TERMS] if abs(e) > 1e-3]

    highlights = []
    peak = float(np.max(np.abs(effects)))
    if peak > 0:
        for w, e in zip(words, effects):
            if abs(e) < max(0.05, 0.12 * peak):
                continue
            for s, end in occurrences[w]:
                highlights.append({"start": _utf16(text, s), "end": _utf16(text, end),
                                   "polarity": "pos" if e > 0 else "neg",
                                   "strength": round(float(abs(e)) / peak, 2)})
        highlights.sort(key=lambda h: h["start"])
    return terms, highlights


def _predict(texts, explain=False):
    probas = pipeline.predict_proba(texts)
    results = []
    for text, p in zip(texts, probas):
        k = int(np.argmax(p))
        r = {
            "sentiment": _classes[k],
            "confidence": round(float(p[k]) * 100, 1),
            "probabilities": {c: round(float(v) * 100, 1) for c, v in zip(_classes, p)},
        }
        if explain:
            r["terms"], r["highlights"] = _explain(text)
        results.append(r)
    return results


def _validate_text(value):
    if not isinstance(value, str) or not value.strip():
        return None, "Empty review text"
    if len(value) > MAX_REVIEW_CHARS:
        return None, f"Review is too long (max {MAX_REVIEW_CHARS} characters)"
    return value.strip(), None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", reviews=SAMPLE_REVIEWS, model_info=MODEL_INFO)


@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return jsonify({"error": "Model unavailable"}), 503
    data = request.get_json(silent=True) or {}
    text, err = _validate_text(data.get("review"))
    if err:
        return jsonify({"error": err}), 400
    return jsonify(_predict([text], explain=True)[0])


@app.route("/predict_all", methods=["POST"])
def predict_all():
    if pipeline is None:
        return jsonify({"error": "Model unavailable"}), 503
    data = request.get_json(silent=True) or {}
    reviews = data.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        return jsonify({"error": "No reviews provided"}), 400
    if len(reviews) > MAX_BATCH:
        return jsonify({"error": f"Too many reviews (max {MAX_BATCH})"}), 400
    texts = []
    for r in reviews:
        text, err = _validate_text(r)
        if err:
            return jsonify({"error": err}), 400
        texts.append(text)
    return jsonify({"results": _predict(texts)})


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))
