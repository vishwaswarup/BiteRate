"""Train the sentiment model.

1. Fit on the training split and report metrics on the frozen held-out test split.
2. Refit on train+test and save to model.pkl for the app.

Run `python data.py` first if data/train.csv does not exist.
"""
import json
import pickle

import pandas as pd
from sklearn.metrics import classification_report, f1_score

from data import load_split
from model import build_pipeline

train, test = load_split()

pipe = build_pipeline().fit(train.text, train.label)
pred = pipe.predict(test.text)
print(classification_report(test.label, pred, digits=3))
print(f"macro-F1 (all): {f1_score(test.label, pred, average='macro'):.3f}")
for src, d in test.assign(pred=pred).groupby("source"):
    print(f"{src}: accuracy {(d.label == d.pred).mean():.3f} (n={len(d)})")

# Final model for deployment uses all the data.
full = pd.concat([train, test], ignore_index=True)
final = build_pipeline().fit(full.text, full.label)
with open("model.pkl", "wb") as f:
    pickle.dump(final, f)
report = classification_report(test.label, pred, output_dict=True)
metrics = {
    "macro_f1": round(report["macro avg"]["f1-score"], 3),
    "accuracy": round(report["accuracy"], 3),
    "per_class": {c: {"precision": round(report[c]["precision"], 3),
                      "recall": round(report[c]["recall"], 3),
                      "f1": round(report[c]["f1-score"], 3),
                      "support": int(report[c]["support"])}
                  for c in ("Positive", "Neutral", "Negative")},
    "train_size": len(full),
    "test_size": len(test),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved model.pkl, classes:", list(final.classes_))
