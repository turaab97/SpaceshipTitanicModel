# train_ensemble.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# 1) Feature‐engineering function (same as before)
def fe(df):
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    df['TotalSpend']   = df[spend_cols].sum(axis=1)
    df['SpendBucket']  = pd.qcut(df['TotalSpend'], 4, labels=False, duplicates='drop')
    df['GroupID']      = df.index.to_series().str.split('_').str[0]
    df['FamilySize']   = df.groupby('GroupID')['GroupID'].transform('count')
    df['AgeBin']       = pd.cut(df['Age'], bins=[0,18,35,60,200],
                                labels=['child','young','adult','senior'])
    return df

# 2) Load & featurize
df = pd.read_csv('train_cleaned.csv', index_col='PassengerId')
df = fe(df)

# 3) Prepare X/y and one-hot encode
cat_cols = ['HomePlanet','Destination','Deck','Side','SpendBucket','AgeBin']
X = pd.get_dummies(df.drop(columns=['Transported','GroupID']), columns=cat_cols, drop_first=True)
y = df['Transported'].astype(int)

# 4) Split for hold-out evaluation
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5) Fit base learners
lgb = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01477,
    num_leaves=42,
    min_child_samples=18,
    subsample=0.672,
    colsample_bytree=0.778,
    reg_alpha=1.43e-08,
    reg_lambda=0.00146,
    random_state=42
)
hgb = HistGradientBoostingClassifier(
    max_iter=1000,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)

lgb.fit(X_tr, y_tr)
hgb.fit(X_tr, y_tr)

# 6) Create meta-features on the hold-out
preds_lgb = lgb.predict_proba(X_va)[:,1]
preds_hgb = hgb.predict_proba(X_va)[:,1]

meta_va = pd.DataFrame({
    'lgb': preds_lgb,
    'hgb': preds_hgb
}, index=X_va.index)

# 7) Train meta-learner
meta_clf = LogisticRegression(max_iter=1000, solver='saga')
meta_clf.fit(meta_va, y_va)

# 8) Evaluate ensemble
meta_pred = meta_clf.predict(meta_va)
meta_prob = meta_clf.predict_proba(meta_va)[:,1]

print("=== Ensemble on Hold-Out ===")
print(classification_report(y_va, meta_pred))
print("Ensemble ROC-AUC:", roc_auc_score(y_va, meta_prob))

# 9) Retrain everything on full data & save submission
#   (you’d repeat fe() on test, encode same columns, predict with base+meta)

