#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spaceship Titanic – Imputation + Feature Engineering + Stacking (updated)
"""

import warnings, os
warnings.filterwarnings("ignore")

# -------------------------------------------------- #
# Basic dependencies
# -------------------------------------------------- #
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OneHotEncoder, StandardScaler,
                                   FunctionTransformer)
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier  # ✅ Added for updated stacking
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------- #
# 1. Utility functions
# -------------------------------------------------- #
def split_cabin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Cabin" in out.columns:
        tmp = out["Cabin"].str.split("/", expand=True)
        out["Deck"] = tmp[0]
        out["Num"] = pd.to_numeric(tmp[1], errors="coerce")
        out["Side"] = tmp[2]
        out.drop(columns="Cabin", inplace=True)
    return out

def derive_id_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["GroupId"] = out.index.str.split("_").str[0]
    out["MemberNumber"] = out.index.str.split("_").str[1]
    out["GroupSize"] = out.groupby("GroupId")["GroupId"].transform("count")
    out["IsAlone"] = (out["GroupSize"] == 1).astype(int)
    return out

# -------------------------------------------------- #
# 2. Load data and split
# -------------------------------------------------- #
train_raw = pd.read_csv("train.csv", index_col="PassengerId")
test_raw = pd.read_csv("test.csv", index_col="PassengerId")

both = (
    pd.concat([train_raw.assign(__split="train"),
               test_raw.assign(__split="test")], axis=0)
      .pipe(split_cabin)
      .pipe(derive_id_feats)
)

# -------------------------------------------------- #
# 3. Cluster-aware imputation
# -------------------------------------------------- #
num_cols = ["Age","RoomService","FoodCourt","ShoppingMall","Spa",
            "VRDeck","Num","GroupSize","IsAlone"]
cat_cols = ["HomePlanet","CryoSleep","Destination","VIP",
            "Deck","Side","MemberNumber"]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("cast", FunctionTransformer(lambda X: X.astype(str), validate=False)),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])
preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])
X_prep = preprocess.fit_transform(both)

def best_k(X, kmin=3, kmax=10):
    best, best_score = None, -np.inf
    for k in range(kmin, kmax+1):
        lab = KMeans(k, n_init="auto", random_state=42).fit_predict(X)
        s = silhouette_score(X, lab)
        print(f"k={k:<2d}  silhouette={s:.4f}")
        if s > best_score: best, best_score = k, s
    print("Optimal k =", best, "\n")
    return best
k_opt = best_k(X_prep)

def impute_cluster(df, k):
    X = preprocess.transform(df)
    labels = KMeans(k, n_init="auto", random_state=42).fit_predict(X)
    tmp = df.copy(); tmp["__cluster"] = labels
    for col in num_cols:
        tmp[col] = tmp.groupby("__cluster")[col].transform(
            lambda s: s.fillna(s.median()))
    for col in cat_cols:
        tmp[col] = tmp.groupby("__cluster")[col].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else np.nan))
    tmp[num_cols] = SimpleImputer(strategy="median").fit_transform(tmp[num_cols])
    tmp[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(tmp[cat_cols])
    return tmp.drop(columns="__cluster")

both = impute_cluster(both, k_opt)
train = both.query("__split=='train'").drop(columns="__split")
test = both.query("__split=='test'").drop(columns="__split")

# -------------------------------------------------- #
# 4. Feature Engineering
# -------------------------------------------------- #
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["CabinDeck"] = out["Deck"].fillna("Unknown")
    out["CabinSide"] = out["Side"].fillna("Unknown")
    out["CabinNum"] = out["Num"].fillna(out["Num"].median())

    spend = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
    for c in spend: out[c] = out[c].fillna(0)
    for c in spend: out[f"log_{c}"] = np.log1p(out[c])
    out["TotalSpend"] = out[spend].sum(axis=1)
    out["SpendPerPerson"] = out["TotalSpend"] / (out["GroupSize"] + 1)

    out["Age"] = out["Age"].fillna(out["Age"].median())
    out["AgeBin"] = pd.cut(out["Age"], bins=[0,12,18,35,60,np.inf],
                           labels=["Child","Teen","Adult","Middle","Senior"],
                           include_lowest=True)

    out["DeckSide"] = out["CabinDeck"] + "_" + out["CabinSide"]

    for c in ["CryoSleep","VIP"]:
        out[c] = out[c].map({True:1, False:0, "True":1, "False":0})\
                        .fillna(0).astype(int)

    return out

train = engineer(train)
test = engineer(test)

cluster_feats = ["CabinNum","log_RoomService","log_FoodCourt",
                 "log_ShoppingMall","log_Spa","log_VRDeck","TotalSpend"]
km = KMeans(n_clusters=6, random_state=42).fit(train[cluster_feats])
train["Cluster"] = km.labels_
test["Cluster"] = km.predict(test[cluster_feats])

spend_cols = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
train["MaxSpendItem"] = train[spend_cols].idxmax(axis=1)
test["MaxSpendItem"] = test[spend_cols].idxmax(axis=1)

for col in ["HomePlanet","Destination","DeckSide"]:
    freq = train[col].value_counts() / len(train)
    train[f"{col}_Freq"] = train[col].map(freq).fillna(0)
    test[f"{col}_Freq"] = test[col].map(freq).fillna(0)

# -------------------------------------------------- #
# 5. Model input prep
# -------------------------------------------------- #
y = train["Transported"].astype(int)

drop_cols = ["Transported","Name","GroupId","HomePlanet","Destination"]
X = train.drop(columns=[c for c in drop_cols if c in train.columns])
X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])

known_cat = ["AgeBin","CabinDeck","CabinSide","DeckSide",
             "MaxSpendItem","Cluster"]
for c in known_cat:
    X[c] = X[c].astype("category").cat.codes
    X_test[c] = X_test[c].astype("category").cat.codes

obj_cols = X.select_dtypes(include="object").columns
for col in obj_cols:
    cats = pd.concat([X[col], X_test[col]]).astype("category").cat.categories
    X[col] = pd.Categorical(X[col], categories=cats).codes
    X_test[col] = pd.Categorical(X_test[col], categories=cats).codes

# -------------------------------------------------- #
# 6. Stacking model (Updated)
# -------------------------------------------------- #

# ✅ New base models
hgb1 = HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=500, max_leaf_nodes=31, random_state=42)
hgb2 = HistGradientBoostingClassifier(
    learning_rate=0.10, max_iter=300, max_leaf_nodes=31, random_state=24)
lgbm = LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=0)

# ✅ Replaced LogisticRegression with CatBoostClassifier for better meta-learning
stack = StackingClassifier(
    estimators=[
        ("hgb1", hgb1),
        ("hgb2", hgb2),
        ("lgbm", lgbm)
    ],
    final_estimator=CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        verbose=False,
        random_state=42
    ),
    cv=5,
    passthrough=True,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(stack, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"Stacking CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# -------------------------------------------------- #
# 7. Train and output
# -------------------------------------------------- #
stack.fit(X, y)
preds = stack.predict(X_test).astype(bool)

pd.DataFrame({
    "PassengerId": X_test.index,
    "Transported": preds
}).to_csv("submission_stack_fusion0606.csv", index=False)

print("\u2714 Training finished – submission_stack_fusion0606.csv generated.")

