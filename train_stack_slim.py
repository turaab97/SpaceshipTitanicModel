# train_stack_slim.py

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score
from itertools import product

# 1) Define spend columns
spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']

# 2) Manual target encoding with smoothing
def target_encode(tr_series, tr_target, val_series, smoothing=10):
    df = pd.DataFrame({'cat': tr_series, 'y': tr_target})
    agg = df.groupby('cat')['y'].agg(['mean','count'])
    global_mean = tr_target.mean()
    agg['smooth'] = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
    return val_series.map(agg['smooth']).fillna(global_mean)

# 3) Load & clean / feature-engineer
def clean_and_feature(df, is_train=True):
    df = df.copy()
    df['HomePlanet']  = df['HomePlanet'].fillna('Unknown')
    df['Destination'] = df['Destination'].fillna('Unknown')
    df['CryoSleep']   = df['CryoSleep'].fillna(False).astype(bool)
    df['VIP']         = df['VIP'].fillna(False).astype(bool)
    df['Title']       = df['Name'].str.extract(r',\s*([^\.]+)\.').fillna('Unknown')
    df['Cabin']       = df['Cabin'].fillna('Unknown/0/Unknown')
    cab = df['Cabin'].str.split('/', expand=True)
    df['Deck']        = cab[0]
    df['CabinNum']    = pd.to_numeric(cab[1], errors='coerce').fillna(0).astype(int)
    df['Side']        = cab[2]
    for c in spend_cols:
        df[c] = df[c].fillna(0.0)
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    for c in spend_cols:
        df[f'{c}_ratio'] = df[c] / (df['TotalSpend'] + 1e-9)
    df['SpendBucket']  = pd.qcut(df['TotalSpend'], 4, labels=False, duplicates='drop')
    df['GroupID']      = df.index.to_series().str.split('_').str[0]
    df['FamilySize']   = df.groupby('GroupID')['GroupID'].transform('count')
    df['IsAlone']      = (df['FamilySize']==1).astype(int)
    df['Route']        = df['HomePlanet'] + "_" + df['Destination']
    df['Age']          = df['Age'].fillna(df['Age'].median())
    df['AgeBin']       = pd.cut(df['Age'], bins=[0,18,35,60,200],
                                labels=['child','young','adult','senior'])
    df = df.drop(columns=['Name','Cabin'])
    if is_train:
        y = df['Transported'].astype(int)
        return df.drop(columns=['Transported']), y
    else:
        return df

# 4) Load raw data
train_raw  = pd.read_csv('train.csv', index_col='PassengerId')
test_raw   = pd.read_csv('test.csv',  index_col='PassengerId')
X_raw, y   = clean_and_feature(train_raw, is_train=True)
X_test_raw = clean_and_feature(test_raw,  is_train=False)

# 5) Adversarial Validation: drop shifting features
Xa = pd.concat([X_raw, X_test_raw], axis=0)
ya = np.concatenate([np.zeros(len(X_raw)), np.ones(len(X_test_raw))])
adv = RandomForestClassifier(n_estimators=200, random_state=42)
adv.fit(Xa.fillna(0), ya)
imp = pd.Series(adv.feature_importances_, index=Xa.columns).nlargest(10)
X_raw.drop(columns=imp.index, inplace=True)
X_test_raw.drop(columns=imp.index, inplace=True)

# 6) Prepare encoding
numerics     = ['CabinNum','Age','TotalSpend','FamilySize'] + [f'{c}_ratio' for c in spend_cols]
categoricals = ['HomePlanet','Destination','CryoSleep','VIP',
                'Title','Deck','Side','SpendBucket','AgeBin','IsAlone','Route']
high_card    = ['Title','Deck','Route']
low_card     = [c for c in categoricals if c not in high_card]

ohe = OneHotEncoder(drop='first', sparse_output=False)
ohe.fit(X_raw[low_card])
X_ohe      = pd.DataFrame(ohe.transform(X_raw[low_card]), index=X_raw.index,
                         columns=ohe.get_feature_names_out(low_card))
X_test_ohe = pd.DataFrame(ohe.transform(X_test_raw[low_card]), index=X_test_raw.index,
                         columns=ohe.get_feature_names_out(low_card))

X_base      = pd.concat([X_raw[numerics + high_card], X_ohe], axis=1)
X_test_base = pd.concat([X_test_raw[numerics + high_card], X_test_ohe], axis=1)
groups      = X_raw['GroupID']

scaler = StandardScaler()
X_base[numerics]      = scaler.fit_transform(X_base[numerics])
X_test_base[numerics] = scaler.transform(X_test_base[numerics])

# 7) Define base learners
models = {
    'hgb': HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.05,
                                          max_leaf_nodes=31, random_state=42),
    'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}

n_train  = len(X_base)
n_test   = len(X_test_base)
oof_preds = {k: np.zeros(n_train) for k in models}
test_preds= {k: np.zeros(n_test)  for k in models}

# 8) Stacking with fold-wise target encoding
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for tr, va in sgkf.split(X_base, y, groups):
    X_tr = X_base.iloc[tr].copy()
    X_va = X_base.iloc[va].copy()
    X_te = X_test_base.copy()
    for col in high_card:
        X_tr[col] = target_encode(X_tr[col], y.iloc[tr], X_tr[col])
        X_va[col] = target_encode(X_tr[col], y.iloc[tr], X_va[col])
        X_te[col] = target_encode(X_tr[col], y.iloc[tr], X_te[col])
    for name, mdl in models.items():
        mdl.fit(X_tr, y.iloc[tr])
        oof_preds[name][va]   = mdl.predict_proba(X_va)[:,1]
        test_preds[name]      += mdl.predict_proba(X_te)[:,1] / 5

# 9) Meta-learner & calibration
meta_X  = pd.DataFrame(oof_preds, index=X_base.index)
base_lr = LogisticRegression(solver='saga', max_iter=2000)
base_lr.fit(meta_X, y)
calib   = CalibratedClassifierCV(base_lr, cv='prefit')
calib.fit(meta_X, y)

# OOF ROC-AUC
meta_oof_prob = calib.predict_proba(meta_X)[:,1]
print("OOF ROC-AUC:", roc_auc_score(y, meta_oof_prob))

# 10) Threshold optimization
best_thr, best_f1 = 0.5, 0
for thr in np.linspace(0.3,0.7,41):
    preds = (meta_oof_prob > thr).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
print("Best threshold:", best_thr)

# 11) Ensemble weight search
best_w, best_auc = None, 0
oof_df = pd.DataFrame(oof_preds)
for w in product(np.linspace(0,1,11), repeat=len(models)):
    if abs(sum(w)-1) > 1e-6: continue
    auc = roc_auc_score(y, (oof_df * w).sum(axis=1))
    if auc > best_auc:
        best_auc, best_w = auc, w
print("Best weights:", best_w)

# 12) Final submission
test_df    = pd.DataFrame(test_preds)
final_prob = (test_df * best_w).sum(axis=1)
submission = pd.DataFrame({'PassengerId': X_test_base.index,
                           'Transported': final_prob > best_thr})
submission.to_csv('submission.csv', index=False)
print("✅ Done! threshold:", best_thr, "weights:", best_w)

