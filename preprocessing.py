import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- NLTK setup ---
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
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
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "]+"
)

# --- Step 2: Preprocessing ---
def clean_text(text):
    # Step 1: Replace common negations
    text = re.sub(r"n't", ' not', text)  # e.g., don't → do not
    
    # Step 2: Convert to lowercase (emojis are case-insensitive)
    text = text.lower()
    
    # Step 3: Add spaces around emojis so they become separate tokens
    text = EMOJI_PATTERN.sub(lambda m: f' {m.group()} ', text)
    
    # Step 4: Remove digits and punctuation but keep letters, spaces, and emojis
    # Instead of removing emojis, we'll rebuild the string keeping only valid chars
    cleaned_chars = []
    for char in text:
        if char.isalpha() or char.isspace():
            # Keep letters and spaces
            cleaned_chars.append(char)
        elif EMOJI_PATTERN.match(char):
            # Keep emojis
            cleaned_chars.append(char)
        else:
            # Replace other characters with space
            cleaned_chars.append(' ')
    text = ''.join(cleaned_chars)
    
    # Step 5: Tokenize - split on whitespace to preserve emojis
    tokens = text.split()
    
    # Step 6: Process tokens: separate regular words from emojis
    processed_tokens = []
    for word in tokens:
        if EMOJI_PATTERN.match(word):
            # Keep emoji as-is
            processed_tokens.append(word)
        elif word not in stop_words and word.isalpha():
            # Lemmatize regular words only
            processed_tokens.append(lemmatizer.lemmatize(word))
    
    # Step 7: Merge negation + next word
    final_tokens = []
    skip_next = False
    for i, word in enumerate(processed_tokens):
        if skip_next:
            skip_next = False
            continue
        if word == 'not' and i+1 < len(processed_tokens) and not EMOJI_PATTERN.match(processed_tokens[i+1]):
            final_tokens.append(f'not_{processed_tokens[i+1]}')
            skip_next = True
        else:
            final_tokens.append(word)
    
    return ' '.join(final_tokens)