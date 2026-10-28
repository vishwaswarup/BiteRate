from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from preprocessing import clean_text

# Emojis are kept as separate tokens by the word vectorizer.
TOKEN_PATTERN = r"(?u)\b\w+\b|[\U0001F300-\U0001FAFF\U00002600-\U000027BF]"


def build_pipeline():
    """Raw review text -> [Negative, Neutral, Positive] probabilities.

    Two views of the text are combined: cleaned/lemmatized word 1-2grams (with
    negation marking) and raw character 2-5grams, which are robust to typos
    and slang.
    """
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2,
                                 max_df=0.9, sublinear_tf=True, token_pattern=TOKEN_PATTERN)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True, max_features=150000)),
    ])
    clf = LogisticRegression(C=1, class_weight="balanced", max_iter=3000)
    return Pipeline([("features", features), ("clf", clf)])
