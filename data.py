"""Build the merged, cleaned 3-class dataset and a frozen train/test split.

Sources:
  yelp   - Restaurant_Reviews.tsv (binary Liked 0/1; no neutral examples)
  zomato - data/zomato_reviews.csv (1-5 star ratings)

Run `python data.py` to (re)create data/train.csv and data/test.csv.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
LABELS = ["Negative", "Neutral", "Positive"]


def _stars_to_label(r):
    if r <= 2:
        return "Negative"
    if r < 4:
        return "Neutral"
    return "Positive"


def load_yelp():
    df = pd.read_csv(os.path.join(HERE, "Restaurant_Reviews.tsv"),
                     delimiter="\t", quoting=3, encoding="utf-8-sig")
    df["label"] = df["Liked"].map({0: "Negative", 1: "Positive"})
    df["source"] = "yelp"
    return df[["Review", "label", "source"]].rename(columns={"Review": "text"})


def load_zomato():
    df = pd.read_csv(os.path.join(DATA_DIR, "zomato_reviews.csv"), encoding="utf-8")
    df["rating"] = pd.to_numeric(df["Rating"], errors="coerce")  # drops the stray "Like" row
    df = df.dropna(subset=["rating", "Review"])
    df["label"] = df["rating"].apply(_stars_to_label)
    df["source"] = "zomato"
    return df[["Review", "label", "source"]].rename(columns={"Review": "text"})


def build(seed=42, test_size=0.2):
    df = pd.concat([load_yelp(), load_zomato()], ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 3]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    strata = df["source"] + "_" + df["label"]
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strata)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def load_split():
    return (pd.read_csv(os.path.join(DATA_DIR, "train.csv")),
            pd.read_csv(os.path.join(DATA_DIR, "test.csv")))


if __name__ == "__main__":
    train, test = build()
    train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    for name, d in (("train", train), ("test", test)):
        print(name, len(d))
        print(pd.crosstab(d["source"], d["label"]))
