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

# --- Step 2: Preprocessing ---
def clean_text(text):
    # Replace common negations
    text = text.lower()
    text = re.sub(r"n't", ' not', text)  # e.g., don't → do not
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Merge negation + next word (optional, simple version)
    new_tokens = []
    skip_next = False
    for i, word in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if word == 'not' and i+1 < len(tokens):
            new_tokens.append(f'not_{tokens[i+1]}')
            skip_next = True
        else:
            new_tokens.append(word)
    
    return ' '.join(new_tokens)