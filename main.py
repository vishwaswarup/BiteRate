import pandas as pd
from preprocessing import clean_text
from model import train_and_evaluate
import pickle

# --- Step 1: Load dataset ---
df = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

# --- Preprocess ---
df['cleaned'] = df['Review'].apply(clean_text)

# --- Train & Evaluate ---
vectorizer, models = train_and_evaluate(df)

best_model = models['Naive Bayes']

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Model and vectorizer saved!")