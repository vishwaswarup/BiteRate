from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_and_evaluate(df, text_column='cleaned', label_column='Liked'):
    # --- Step 3: Feature Extraction ---
    X = df[text_column]
    y = df[label_column]
    # Updated token_pattern to handle emojis as separate tokens
    # Default pattern \b\w\w+\b would skip emojis, so we use a custom analyzer or adjust pattern
    vectorizer = TfidfVectorizer(
        max_features=5000, 
        ngram_range=(1,2),
        token_pattern=r'(?u)\b\w+\b|[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]'
    )
    X_tfidf = vectorizer.fit_transform(X)

    # --- Step 4: Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # --- Step 5: Define models ---
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # --- Step 6: Train & Evaluate ---
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n=== {name} ===")
        print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return vectorizer, models