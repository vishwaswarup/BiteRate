import pandas as pd
from preprocessing import clean_text
from model import train_and_evaluate

# --- Step 1: Load dataset ---
df = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

# --- Preprocess ---
df['cleaned'] = df['Review'].apply(clean_text)

# --- Train & Evaluate ---
vectorizer, models = train_and_evaluate(df)