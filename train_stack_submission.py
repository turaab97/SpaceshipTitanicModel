# train_stack_submission.py

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from category_encoders import TargetEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report

# --- 1) Load raw data ---
train_raw = pd.read_csv('train.csv', index_col='PassengerId')
test_raw  = pd.read_csv('test.csv',  index_col='PassengerId')

# --- 2) Clean & Feature Engineering ---
def clean_and_feature(df, is_train=True):
    df = df.copy()
    # Fill & cast
    df['HomePlanet']  = df['HomePlanet'].fillna('Unknown')
    df['Destination'] = df['Destination'].fillna('Unknown')
    df['CryoSleep']   = df['CryoSleep'].fillna(False).astype(bool)
    df['VIP']         = df['VIP'].fillna(False).astype(bool)
    # Title from Name
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.').fillna('Unknown')
    # Cabin → Deck, CabinNum, Side
    df['Cabin'] = df['Cabin'].fillna('Unknown/0/Unknown')
    cab_split  = df['Cabin'].str.split('/', expand=True)
    df['Deck']     = cab_split[0]
    df['CabinNum'] = pd.to_numeric(cab_split[1], errors='coerce').fillna(0).astype(int)
    df['Side']     = cab_split[2]
    # Spending features
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    for c in spend_cols:
        df[c] = df[c].fillna(0.0)
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    for c in spend_cols:
        df[f'{c}_ratio'] = df[c] / (df['TotalSpend'] + 1e-9)
    # New features
    df['SpendBucket']   = pd.qcut(df['TotalSpend'], 4, labels=False, duplicates='drop')
    df['GroupID']       = df.index.to_series().str.split('_').str[0]
    df['FamilySize']    = df.groupby('GroupID')['GroupID'].transform('count')
    df['IsAlone']       = (df['FamilySize']==1).astype(int)
    df['Route']         = df['HomePlanet'] + "_" + df['Destination']
    df['Title_Deck']    = df['Title'] + "_" + df['Deck']
    df['CabinNumBin']   = pd.qcut(df['CabinNum'], 5, labels=False, duplicates='drop')
    df['SpendPerCapita']= df['TotalSpend'] / (df['FamilySize'] + 1e-9)
    df['HighSpender']   = (df['SpendPerCapita'] > df['SpendPerCapita'].median()).astype(int)
    df['Age']           = df['Age'].fillna(df['Age'].median())
    df['AgeBin']        = pd.cut(df['Age'], bins=[0,18,35,60,200],
                                 labels=['child','young','adult','senior'])
    df['Age_x_Fam']     = df['Age'] * df['FamilySize']
    # Drop unused
    df = df.drop(columns=['Name','Cabin'])
    # Split off target
    if is_train:
        y = df['Transported'].astype(int)
        return df.drop(columns=['Transported']), y
    else:
        return df

Xr, y = clean_and_feature(train_raw, is_train=True)
Xt    = clean_and_feature(test_raw,  is_train=False)

# Keep groups for CV, then drop
groups = Xr['GroupID'].values
Xr     = Xr.drop(columns=['GroupID'])
Xt     = Xt.drop(columns=['GroupID'])

# --- 3) Define feature lists ---
numerics = (
    ['CabinNum','Age','TotalSpend','FamilySize','SpendPerCapita','Age_x_Fam'] +
    [f'{c}_ratio' for c in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]
)
categoricals = [
    'HomePlanet','Destination','CryoSleep','VIP','Title','Deck','Side',
    'SpendBucket','AgeBin','IsAlone','Title_Deck','CabinNumBin','HighSpender'
]
target_encode_cols = ['Title','Deck','Route']

# --- 4) Target-encode high-cardinality fields ---
te = TargetEncoder(cols=target_encode_cols, smoothing=0.3)
Xr[target_encode_cols] = te.fit_transform(Xr[target_encode_cols], y)
Xt[target_encode_cols] = te.transform(Xt[target_encode_cols])

# --- 5) One-hot encode the rest ---
ohe_cols = [c for c in categoricals if c not in target_encode_cols]
ohe = OneHotEncoder(drop='first', sparse_output=False)
ohe.fit(Xr[ohe_cols])
Xr_ohe = pd.DataFrame(
    ohe.transform(Xr[ohe_cols]),
    index=Xr.index,
    columns=ohe.get_feature_names_out(ohe_cols)
)
Xt_ohe = pd.DataFrame(
    ohe.transform(Xt[ohe_cols]),
    index=Xt.index,
    columns=ohe.get_feature_names_out(ohe_cols)
)

# --- 6) Assemble final matrices ---
X      = pd.concat([Xr[numerics + target_encode_cols], Xr_ohe], axis=1)
X_test = pd.concat([Xt[numerics + target_encode_cols], Xt_ohe], axis=1)

# Scale numerics for MLP & Logistic
scaler = StandardScaler()
X[numerics]      = scaler.fit_transform(X[numerics])
X_test[numerics] = scaler.transform(X_test[numerics])

# --- 7) Define base learners ---
models = {
    'lgb': LGBMClassifier(
        n_estimators=1000, learning_rate=0.0148, num_leaves=42,
        min_child_samples=18, subsample=0.672, colsample_bytree=0.778,
        reg_alpha=1.43e-08, reg_lambda=0.00146, random_state=42
    ),
    'hgb': HistGradientBoostingClassifier(
        max_iter=1000, learning_rate=0.05, max_leaf_nodes=31, random_state=42
    ),
    'cat': CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        eval_metric='AUC', random_seed=42, verbose=False
    ),
    'mlp': MLPClassifier(
        hidden_layer_sizes=(100,), activation='relu',
        max_iter=1000, random_state=42
    )
}

# --- 8) 5-fold StratifiedGroupKFold stacking ---
n_train, n_test = X.shape[0], X_test.shape[0]
oof_preds  = {k: np.zeros(n_train) for k in models}
test_preds = {k: np.zeros(n_test)  for k in models}

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, va_idx in sgkf.split(X, y, groups):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        oof_preds[name][va_idx] = mdl.predict_proba(X_va)[:,1]
        test_preds[name]        += mdl.predict_proba(X_test)[:,1] / 5

# --- 9) Meta-learner (calibrated logistic) ---
meta_X   = pd.DataFrame(oof_preds, index=X.index)
base_clf = LogisticRegression(solver='saga', max_iter=2000)
base_clf.fit(meta_X, y)
meta_clf = CalibratedClassifierCV(base_clf, cv='prefit')
meta_clf.fit(meta_X, y)

# Evaluate OOF
meta_oof_prob = meta_clf.predict_proba(meta_X)[:,1]
print("Stacked OOF ROC-AUC:", roc_auc_score(y, meta_oof_prob))
print(classification_report(y, meta_clf.predict(meta_X)))

# --- 10) Final test predictions & submission ---
meta_test  = pd.DataFrame(test_preds, index=X_test.index)
final_prob = meta_clf.predict_proba(meta_test)[:,1]
submission = pd.DataFrame({
    'PassengerId': X_test.index,
    'Transported': final_prob > 0.5
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv ready – go hit 0.85+!")
