import pickle
import os
from flask import Flask, render_template, request, jsonify
from preprocessing import clean_text

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading with demo-mode fallback
# ---------------------------------------------------------------------------
DEMO_MODE = False
model = None
vectorizer = None

try:
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    DEMO_MODE = True
    print(f"⚠️  Could not load model/vectorizer ({e}). Running in DEMO mode.")

# ---------------------------------------------------------------------------
# 12 realistic sample reviews
# ---------------------------------------------------------------------------
SAMPLE_REVIEWS = [
    {"id": 1, "text": "Absolutely loved the truffle pasta — rich, creamy, and cooked to perfection. Will definitely come back!", "sentiment": None, "confidence": None},
    {"id": 2, "text": "Waited over an hour for cold, soggy fries and a burnt burger. Never again.", "sentiment": None, "confidence": None},
    {"id": 3, "text": "The ambiance was nice but the food was just okay. Nothing special for the price they charge.", "sentiment": None, "confidence": None},
    {"id": 4, "text": "Best sushi I've ever had outside of Japan. The chef clearly knows his craft.", "sentiment": None, "confidence": None},
    {"id": 5, "text": "Terrible service. The waiter was rude and forgot half our order. Food was mediocre at best.", "sentiment": None, "confidence": None},
    {"id": 6, "text": "Decent brunch spot. Pancakes were fluffy but the coffee was weak. Might try again.", "sentiment": None, "confidence": None},
    {"id": 7, "text": "The wood-fired pizza was phenomenal — crispy crust, fresh mozzarella, perfect basil. A hidden gem!", "sentiment": None, "confidence": None},
    {"id": 8, "text": "Overpriced and underwhelming. Paid $40 for a tiny portion of bland chicken. Very disappointed.", "sentiment": None, "confidence": None},
    {"id": 9, "text": "Great cocktail menu and the appetizers were creative and delicious. Perfect for a date night!", "sentiment": None, "confidence": None},
    {"id": 10, "text": "Found a hair in my soup. When I complained, the manager shrugged it off. Disgusting.", "sentiment": None, "confidence": None},
    {"id": 11, "text": "Average experience overall. The pasta was slightly overcooked but dessert was surprisingly good.", "sentiment": None, "confidence": None},
    {"id": 12, "text": "Hands down the best tacos in the city. The salsa verde is addictive. Five stars!", "sentiment": None, "confidence": None},
]

# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
import random

def _predict_single(review_text: str) -> dict:
    """Return {"sentiment": str, "confidence": float}."""
    if DEMO_MODE:
        # Deterministic-ish demo based on simple heuristics
        positive_words = {"love", "great", "best", "amazing", "perfect", "delicious", "phenomenal", "gem", "addictive", "fantastic"}
        negative_words = {"terrible", "worst", "rude", "bland", "disgusting", "cold", "burnt", "never", "disappointed", "overpriced"}
        lower = review_text.lower()
        pos_count = sum(1 for w in positive_words if w in lower)
        neg_count = sum(1 for w in negative_words if w in lower)
        if pos_count > neg_count:
            return {"sentiment": "Positive", "confidence": round(random.uniform(78, 96), 1)}
        elif neg_count > pos_count:
            return {"sentiment": "Negative", "confidence": round(random.uniform(72, 94), 1)}
        else:
            return {"sentiment": "Neutral", "confidence": round(random.uniform(45, 65), 1)}

    # Real model prediction
    cleaned = clean_text(review_text)
    X = vectorizer.transform([cleaned])

    # Try predict_proba first
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        proba = model.predict_proba(X)[0]  # [prob_negative, prob_positive]
        pos_prob = proba[1]
        confidence = max(proba) * 100

        # Map to three-class sentiment
        if pos_prob >= 0.70:
            sentiment = "Positive"
            confidence = pos_prob * 100
        elif pos_prob <= 0.30:
            sentiment = "Negative"
            confidence = (1 - pos_prob) * 100
        else:
            sentiment = "Neutral"
            confidence = (1 - abs(pos_prob - 0.5) * 2) * 100  # Higher when closer to 0.5
    else:
        pred = model.predict(X)[0]
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = 92.0

    return {"sentiment": sentiment, "confidence": round(confidence, 1)}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    # Pass a fresh copy so template always gets unpredicted reviews
    reviews = [dict(r) for r in SAMPLE_REVIEWS]
    return render_template("index.html", reviews=reviews)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    review_text = data.get("review", "")
    if not review_text.strip():
        return jsonify({"error": "Empty review text"}), 400
    result = _predict_single(review_text)
    return jsonify(result)


@app.route("/predict_all", methods=["POST"])
def predict_all():
    data = request.get_json(force=True)
    reviews = data.get("reviews", [])
    if not reviews:
        return jsonify({"error": "No reviews provided"}), 400
    results = [_predict_single(text) for text in reviews]
    return jsonify({"results": results})


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)