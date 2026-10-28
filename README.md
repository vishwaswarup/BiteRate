# BiteRate

Restaurant review sentiment analysis: classifies a review as **Positive**, **Neutral** or **Negative**, shows the probability of each class, and highlights the words that drove the prediction. Flask + scikit-learn, with a dashboard UI.

## Data

Two sources are merged by `data.py` (deduplicated, blank rows dropped) into a frozen, stratified 80/20 split in `data/train.csv` and `data/test.csv`:

| Source | Rows (after cleaning) | Labels |
|---|---|---|
| `Restaurant_Reviews.tsv` (Kaggle, short reviews) | ~1,000 | Liked 0/1 → Negative / Positive |
| `data/zomato_reviews.csv` (Zomato, long reviews) | ~9,300 | 1-2★ Negative, 2.5-3.5★ Neutral, 4-5★ Positive |

Neutral examples come only from the star-rated source. The test set is never used for training when reporting metrics.

## Model

- **Preprocessing** (`preprocessing.py`): contraction expansion (including curly apostrophes), lemmatization, stopword removal that *keeps* negations and intensifiers, negation marking limited to the same clause (`not good` → `not_good`), and emojis as individual tokens.
- **Features** (`model.py`): TF-IDF word 1-2-grams on the cleaned text + TF-IDF character 2-5-grams on the raw text.
- **Classifier**: class-balanced multinomial logistic regression (probabilities power the UI).

### Held-out results (2,075 reviews)

| | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.82 | 0.87 | 0.85 |
| Neutral | 0.42 | 0.48 | 0.45 |
| Positive | 0.93 | 0.88 | 0.91 |

Macro-F1 **0.73**, accuracy **0.83** (0.87 on the short Yelp-style reviews, 0.83 on Zomato). Neutral is the hardest class, as expected for "mixed" reviews and noisy star labels. These numbers are for a model trained on the training split only; the deployed `model.pkl` is refit on all data.

## Run it

```bash
pip install -r requirements-train.txt   # training + app (the deployed app only needs requirements.txt)
python data.py                          # rebuild the split (only if data changes)
python main.py                          # train, print held-out metrics, write model.pkl + metrics.json
python app.py                           # http://localhost:5000
python -m unittest test_preprocessing   # preprocessing tests
```

On macOS, port 5000 may be taken by AirPlay Receiver; change the port in `app.py` if so. Set `FLASK_DEBUG=1` for debug mode.

## API

- `POST /predict` `{"review": "..."}` → `{sentiment, confidence, probabilities, terms}`
- `POST /predict_all` `{"reviews": ["...", ...]}` (max 100) → `{results: [...]}`

Input is limited to 5,000 characters per review. If `model.pkl` can't be loaded the API returns 503 rather than guessing.

## Deploying

`vercel.json` is configured for Vercel. NLTK data is bundled in `nltk_data/`, so nothing is downloaded at runtime. `scikit-learn` is pinned to the version used to train `model.pkl`; retrain if you change it.
