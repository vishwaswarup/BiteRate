from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False
import numpy as np

def train_and_evaluate(df, text_column='cleaned', label_column='Liked'):
    # --- Step 3: Feature Extraction ---
    X = df[text_column]
    y = df[label_column]
    # Updated token_pattern to handle emojis as separate tokens
    # Default pattern \b\w\w+\b would skip emojis, so we use a custom analyzer or adjust pattern
    vectorizer = TfidfVectorizer(
        max_features=10000, 
        ngram_range=(1,3),
        min_df=2,
        max_df=0.9,
        token_pattern=r'(?u)\b\w+\b|[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]'
    )
    X_tfidf = vectorizer.fit_transform(X)

    # --- Step 4: Cross Validation Setup ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Step 5: Define models ---
    
    # Tuning Logistic Regression
    print("\nTuning Logistic Regression for C in (0.1, 1, 10)...")
    lr_base = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    lr_grid = GridSearchCV(lr_base, {'C': [0.1, 1, 10]}, cv=cv, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_tfidf, y)
    best_lr = lr_grid.best_estimator_
    print(f"Best LR params: {lr_grid.best_params_}")

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression (Tuned)": best_lr,
        "LinearSVC (via SVC)": SVC(kernel='linear', probability=True, random_state=42)
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(eval_metric='logloss', random_state=42)

    # Ensemble: Voting Classifier (Soft Voting)
    ensemble = VotingClassifier(
        estimators=[
            ('lr', best_lr),
            ('nb', MultinomialNB()),
            ('svc', SVC(kernel='linear', probability=True, random_state=42))
        ],
        voting='soft'
    )
    models["Ensemble (LR + NB + SVC)"] = ensemble

    # --- Step 6: Train & Evaluate with CV ---
    results = {}
    best_model_name = None
    best_accuracy = 0.0

    print("\nTraining and Evaluating Models via 5-Fold Stratified CV...")
    for name, model in models.items():
        # Evaluate model using cross_val_score
        scores = cross_val_score(model, X_tfidf, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_acc = scores.mean()
        std_acc = scores.std()
        
        results[name] = mean_acc

        print(f"Model: {name}")
        print(f"Mean Accuracy: {mean_acc * 100:.2f}%")
        print(f"Std Dev: {std_acc * 100:.2f}%\n")
        
        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_model_name = name

        # Fit the model on the full dataset for final deployment
        if name != "Logistic Regression (Tuned)": # LR was already fit fully via grid search
            model.fit(X_tfidf, y)

    print("\n" + "="*40)
    print("MODEL PERFORMANCE COMPARISON (MEAN CV ACCURACY)")
    print("="*40)
    for name, acc in results.items():
        print(f"{name.ljust(30)}: {acc * 100:.2f}%")
    
    print("-" * 40)
    print(f"🌟 BEST MODEL: {best_model_name} with Accuracy {best_accuracy * 100:.2f}% 🌟")
    print("="*40 + "\n")

    return vectorizer, models, best_model_name