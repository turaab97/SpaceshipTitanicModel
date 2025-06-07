# train_stack_group_cv.py

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report

# 1) Load raw data
train_raw = pd.read_csv('train.csv', index_col='PassengerId')
test_raw  = pd.read_csv('test.csv',  index_col='PassengerId')

# 2) Clean & feature‐engineer, keep GroupID for CV
def clean_and_feature(df, is_train=True):
    df = df.copy()
    # fills & casts
    df['HomePlanet']  = df['HomePlanet'].fillna('Unknown')
    df['Destination'] = df['Destination'].fillna('Unknown')
    df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(bool)
    df['VIP']       = df['VIP'].fillna(False).astype(bool)
    # Title
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.').fillna('Unknown')
    # Cabin → Deck, CabinNum, Side
    df['Cabin'] = df['Cabin'].fillna('Unknown/0/Unknown')
    csplit = df['Cabin'].str.split('/', expand=True)
    df['Deck']     = csplit[0]
    df['CabinNum'] = pd.to_numeric(csplit[1], errors='coerce').fillna(0).astype(int)
    df['Side']     = csplit[2]
    # Spending
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    for c in spend_cols:
        df[c] = df[c].fillna(0.0)
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    for c in spend_cols:
        df[f'{c}_ratio'] = df[c] / (df['TotalSpend'] + 1e-9)
    df['SpendBucket'] = pd.qcut(df['TotalSpend'], 4, labels=False, duplicates='drop')
    # Family
    df['GroupID']    = df.index.to_series().str.split('_').str[0]
    df['FamilySize'] = df.groupby('GroupID')['GroupID'].transform('count')
    df['IsAlone']    = (df['FamilySize']==1).astype(int)
    # Age
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,18,35,60,200],
                         labels=['child','young','adult','senior'])
    # drop raw
    df = df.drop(columns=['Name','Cabin'])
    # split y if train
    if is_train:
        y = df['Transported'].astype(int)
        return df.drop(columns=['Transported']), y
    else:
        return df

Xr, y = clean_and_feature(train_raw, is_train=True)
Xt     = clean_and_feature(test_raw,  is_train=False)

groups = Xr['GroupID'].values            # for group‐aware CV
Xr      = Xr.drop(columns=['GroupID'])   # drop before encoding
Xt      = Xt.drop(columns=['GroupID'])

# 3) Encode
cat_cols = [
    'HomePlanet','Destination','CryoSleep','VIP',
    'Title','Deck','Side','SpendBucket','AgeBin','IsAlone'
]
num_cols = ['CabinNum','Age','TotalSpend','FamilySize'] + \
           [f'{c}_ratio' for c in
            ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]

enc = OneHotEncoder(drop='first', sparse_output=False)
enc.fit(Xr[cat_cols])

def encode(df):
    ohe = pd.DataFrame(
        enc.transform(df[cat_cols]),
        index=df.index,
        columns=enc.get_feature_names_out(cat_cols)
    )
    return pd.concat([df[num_cols], ohe], axis=1)

X, X_test = encode(Xr), encode(Xt)

# 4) Prepare OOF / test‐pred arrays
n, m = X.shape[0], X_test.shape[0]
oof = {'lgb':np.zeros(n), 'hgb':np.zeros(n), 'cat':np.zeros(n)}
preds = {'lgb':np.zeros(m), 'hgb':np.zeros(m), 'cat':np.zeros(m)}

# 5) Define models
lgb = LGBMClassifier(
    n_estimators=1000, learning_rate=0.0148, num_leaves=42,
    min_child_samples=18, subsample=0.672, colsample_bytree=0.778,
    reg_alpha=1.43e-08, reg_lambda=0.00146, random_state=42
)
hgb = HistGradientBoostingClassifier(
    max_iter=1000, learning_rate=0.05, max_leaf_nodes=31, random_state=42
)
cat = CatBoostClassifier(
    iterations=1000, learning_rate=0.05, depth=6,
    eval_metric='AUC', random_seed=42, verbose=False
)

# 6) StratifiedGroupKFold stacking
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, va_idx in sgkf.split(X, y, groups):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

    for name, model in (('lgb', lgb), ('hgb', hgb), ('cat', cat)):
        model.fit(Xtr, ytr)
        oof[name][va_idx]    = model.predict_proba(Xva)[:,1]
        preds[name]         += model.predict_proba(X_test)[:,1] / 5

# 7) Meta‐learner
meta_X = pd.DataFrame(oof, index=X.index)
meta_clf = LogisticRegression(solver='saga', max_iter=2000)
meta_clf.fit(meta_X, y)

# OOF eval
meta_oof = meta_clf.predict_proba(meta_X)[:,1]
print("Meta OOF ROC-AUC:", roc_auc_score(y, meta_oof))
print(classification_report(y, meta_clf.predict(meta_X)))

# 8) Final submission
meta_test = pd.DataFrame(preds, index=X_test.index)
final = meta_clf.predict(meta_test)
sub = pd.DataFrame({'PassengerId': X_test.index, 'Transported': final.astype(bool)})
sub.to_csv('submission.csv', index=False)
print("✅ submission.csv ready — this time using GROUP‐aware CV")

