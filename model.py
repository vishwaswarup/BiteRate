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
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
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