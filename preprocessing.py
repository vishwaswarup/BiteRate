import re
import os
import nltk
from nltk.stem import WordNetLemmatizer

# --- NLTK setup: data is bundled in ./nltk_data so no network is needed at runtime ---
_nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
nltk.data.path.insert(0, _nltk_data_dir)

def _has_nltk_resource(pkg):
    # NLTK finds unzipped data as corpora/<pkg> and bundled archives as corpora/<pkg>.zip/<pkg>/
    for path in (f"corpora/{pkg}", f"corpora/{pkg}.zip/{pkg}/"):
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            pass
    return False


for _pkg in ("stopwords", "wordnet"):
    if not _has_nltk_resource(_pkg):
        nltk.download(_pkg, download_dir=_nltk_data_dir, quiet=True)

from nltk.corpus import stopwords

# Words that carry sentiment and must survive stopword removal.
NEGATIONS = {"not", "no", "nor", "never", "none", "nothing", "neither", "without", "hardly", "barely"}
KEEP = NEGATIONS | {"very", "too", "only", "most", "more", "few", "against", "off", "just", "so", "really"}

stop_words = set(stopwords.words("english")) - KEEP
lemmatizer = WordNetLemmatizer()

# Emoji pattern - covers most common emoji ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002600-\U000027BF"  # Misc symbols & dingbats
    "]+"
)

NEGATION_SCOPE = 2  # how many following words a negation applies to
_CLAUSE_BREAK = re.compile(r"[.!?,;:\n]|\bbut\b|\bhowever\b")


def _expand_contractions(text):
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b|\bcannot\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)  # don't, wasn't, isn't, ...
    return text


def _mark_clause(clause):
    """Lemmatize a clause and prefix words in negation scope with not_."""
    clause = EMOJI_PATTERN.sub(lambda m: " " + " ".join(m.group()) + " ", clause)  # one token per emoji
    out, scope = [], 0
    for word in clause.split():
        if EMOJI_PATTERN.fullmatch(word):
            out.append(word)
        elif word in NEGATIONS:
            scope = NEGATION_SCOPE
        elif word.isalpha() and (word not in stop_words) and len(word) > 1:
            word = lemmatizer.lemmatize(word)
            if scope:
                out.append(f"not_{word}")
                scope -= 1
            else:
                out.append(word)
    return out


# --- Step 2: Preprocessing ---
def clean_text(text):
    text = _expand_contractions(str(text).lower())
    # Keep punctuation until clauses are split so negation does not leak across them.
    pieces = re.split(r"(?:[.!?,;:\n]+|\bbut\b|\bhowever\b)", text)
    tokens = []
    for piece in pieces:
        piece = "".join(c if (c.isalpha() or c.isspace() or EMOJI_PATTERN.fullmatch(c)) else " " for c in piece)
        tokens.extend(_mark_clause(piece))
    return " ".join(tokens)
