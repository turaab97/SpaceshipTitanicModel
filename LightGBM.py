#accuracy of 0.80500

#Clean & fill

# Clean & fill
#   - Missing planet/destination → “Unknown”
#   - CryoSleep/VIP blanks → treated as False
#   - Extract your title (Mr/Ms/Dr) and split the cabin string into deck, number, side
#   - Zero-fill all spending columns
#
# Cook up features
#   - Total spend + ratios per service
#   - Quartile bucket of spend
#   - Family groups (size, “is alone?”) and group-level stats (max/min/std of age & spend)
#   - Interactions: Age×Spend, VIP×CryoSleep, even/odd cabin number, missing-data counts, etc.
#
# Drop leaky features
#   - Train a quick classifier to tell train vs test rows apart
#   - Yank out the top handful of features that shift most between train and test
#
# Encode cats
#   - One-hot encode low-cardinality categories (HomePlanet, Destination, etc.)
#   - K-fold target-encode high-card fields (Title, Deck, Route) inside each CV fold to avoid peeking
#
# Scale numbers
#   - Standardize all numeric columns (zero mean, unit variance)
#
# Fit six different models
#   - scikit-learn’s HistGradientBoosting
#   - A simple MLP neural net
#   - LightGBM, CatBoost, XGBoost
#   - ExtraTrees
#   (5-fold “StratifiedGroup” split so families never leak)
#
# Stack ’em
#   - Collect each base model’s out-of-fold probabilities into a new tiny dataset
#   - Train a LightGBM meta-model on that to learn how to blend them
#
# Fine-tune the finish
#   - Scan thresholds (0.3–0.7) to pick the best cut-off for “yes/no”
#   - Grid-search simple weights (that sum to 1) to optimally blend the six base model probs

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score
from itertools import product
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

pd.set_option('future.no_silent_downcasting', True)

spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']

def kfold_target_encode(tr_series, tr_target, val_series, n_splits=5, smoothing=10, random_state=42):
    global_mean = tr_target.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = pd.Series(np.nan, index=tr_series.index)
    for tr_idx, val_idx in kf.split(tr_series):
        fold_tr, fold_val = tr_series.iloc[tr_idx], tr_series.iloc[val_idx]
        fold_target = tr_target.iloc[tr_idx]
        df = pd.DataFrame({'cat': fold_tr, 'y': fold_target})
        agg = df.groupby('cat')['y'].agg(['mean','count'])
        agg['smooth'] = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
        oof.iloc[val_idx] = fold_val.map(agg['smooth']).fillna(global_mean)
    df = pd.DataFrame({'cat': tr_series, 'y': tr_target})
    agg = df.groupby('cat')['y'].agg(['mean','count'])
    agg['smooth'] = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
    val_encoded = val_series.map(agg['smooth']).fillna(global_mean)
    return oof, val_encoded

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
        df[f'{c}_was_missing'] = df[c].isna().astype(int)
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
    # More group-level features
    for c in spend_cols + ['Age']:
        df[f'Group_{c}_max'] = df.groupby('GroupID')[c].transform('max')
        df[f'Group_{c}_min'] = df.groupby('GroupID')[c].transform('min')
        df[f'Group_{c}_std'] = df.groupby('GroupID')[c].transform('std').fillna(0)
    # More interactions
    df['Age_TotalSpend'] = df['Age'] * df['TotalSpend']
    df['CabinNum_even'] = (df['CabinNum'] % 2 == 0).astype(int)
    df['Spend_per_Age'] = df['TotalSpend'] / (df['Age'] + 1)
    df['VIP_CryoSleep'] = df['VIP'].astype(int) * df['CryoSleep'].astype(int)
    # Missingness pattern
    df['MissingCount'] = df.isnull().sum(axis=1)
    df = df.drop(columns=['Name','Cabin'])
    if is_train:
        y = df['Transported'].astype(int)
        return df.drop(columns=['Transported']), y
    else:
        return df

train_raw  = pd.read_csv('train.csv', index_col='PassengerId')
test_raw   = pd.read_csv('test.csv',  index_col='PassengerId')
X_raw, y   = clean_and_feature(train_raw, is_train=True)
X_test_raw = clean_and_feature(test_raw,  is_train=False)

Xa = pd.concat([X_raw, X_test_raw], axis=0)
ya = np.concatenate([np.zeros(len(X_raw)), np.ones(len(X_test_raw))])
for col in Xa.select_dtypes(include='category').columns:
    Xa[col] = Xa[col].astype(str)
for col in X_raw.select_dtypes(include='category').columns:
    X_raw[col] = X_raw[col].astype(str)
for col in X_test_raw.select_dtypes(include='category').columns:
    X_test_raw[col] = X_test_raw[col].astype(str)
Xa = pd.get_dummies(Xa, drop_first=True)
adv = RandomForestClassifier(n_estimators=200, random_state=42)
adv.fit(Xa.fillna(0), ya)
imp = pd.Series(adv.feature_importances_, index=Xa.columns).nlargest(5)
cols_to_drop = [col for col in imp.index if col in X_raw.columns]
X_raw = X_raw.drop(columns=cols_to_drop)
X_test_raw = X_test_raw.drop(columns=cols_to_drop)

numerics     = ['CabinNum','Age','TotalSpend','FamilySize','Age_TotalSpend','Spend_per_Age','VIP_CryoSleep','MissingCount'] + [f'{c}_ratio' for c in spend_cols]
numerics    += [f'Group_{c}_mean' for c in spend_cols] + [f'Group_{c}_max' for c in spend_cols + ['Age']] + [f'Group_{c}_min' for c in spend_cols + ['Age']] + [f'Group_{c}_std' for c in spend_cols + ['Age']] + [f'{c}_was_missing' for c in spend_cols] + ['CabinNum_even']
categoricals = ['HomePlanet','Destination','CryoSleep','VIP','Title','Deck','Side','SpendBucket','AgeBin','IsAlone','Route']
high_card    = ['Title','Deck','Route']
low_card     = [c for c in categoricals if c not in high_card]
numerics = [col for col in numerics if col in X_raw.columns]
high_card = [col for col in high_card if col in X_raw.columns]

ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
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

models = {
    'hgb': HistGradientBoostingClassifier(max_iter=1200, learning_rate=0.04, max_leaf_nodes=35, random_state=42),
    'mlp': MLPClassifier(hidden_layer_sizes=(120,), max_iter=1200, random_state=42),
    'lgb': LGBMClassifier(n_estimators=600, learning_rate=0.04, random_state=42),
    'cat': CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, verbose=0, random_state=42),
    'xgb': XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric='logloss', random_state=42),
    'et' : ExtraTreesClassifier(n_estimators=300, random_state=42)
}

n_train  = len(X_base)
n_test   = len(X_test_base)
oof_preds = {k: np.zeros(n_train) for k in models}
test_preds= {k: np.zeros(n_test)  for k in models}

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for tr, va in sgkf.split(X_base, y, groups):
    X_tr = X_base.iloc[tr].copy()
    X_va = X_base.iloc[va].copy()
    X_te = X_test_base.copy()
    for col in high_card:
        tr_enc, va_enc = kfold_target_encode(X_tr[col], y.iloc[tr], X_va[col])
        _, te_enc = kfold_target_encode(X_tr[col], y.iloc[tr], X_te[col])
        X_tr[col] = tr_enc
        X_va[col] = va_enc
        X_te[col] = te_enc
    for name, mdl in models.items():
        mdl.fit(X_tr, y.iloc[tr])
        oof_preds[name][va]   = mdl.predict_proba(X_va)[:,1]
        test_preds[name]      += mdl.predict_proba(X_te)[:,1] / 5

meta_X  = pd.DataFrame(oof_preds, index=X_base.index)
meta_model = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
meta_model.fit(meta_X, y)
meta_oof_prob = meta_model.predict_proba(meta_X)[:,1]
print("OOF ROC-AUC:", roc_auc_score(y, meta_oof_prob))

best_thr, best_f1 = 0.5, 0
for thr in np.linspace(0.3,0.7,41):
    preds = (meta_oof_prob > thr).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
print("Best threshold:", best_thr)

best_w, best_auc = None, 0
oof_df = pd.DataFrame(oof_preds)
for w in product(np.linspace(0,1,11), repeat=len(models)):
    if abs(sum(w)-1) > 1e-6: continue
    auc = roc_auc_score(y, (oof_df * w).sum(axis=1))
    if auc > best_auc:
        best_auc, best_w = auc, w
print("Best weights:", best_w)

test_df    = pd.DataFrame(test_preds)
final_prob = (test_df * best_w).sum(axis=1)
submission = pd.DataFrame({'PassengerId': X_test_base.index,
                           'Transported': final_prob > best_thr})
submission.to_csv('submission.csv', index=False)
print("✅ Done! threshold:", best_thr, "weights:", best_w)
