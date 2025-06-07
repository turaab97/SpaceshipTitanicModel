import pandas as pd
import numpy as np
import optuna
import shap
from sklearn.model_selection import StratifiedGroupKFold, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.cluster import KMeans
from itertools import product
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
pd.set_option('future.no_silent_downcasting', True)

RANDOM_STATE = 42
N_SPLITS = 5
PSEUDO_LABEL_THRESHOLD = 0.95
spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']

def optimize_lgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': RANDOM_STATE
    }
    model = LGBMClassifier(**params)
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    return score

def kfold_target_encode(tr_series, tr_target, val_series, n_splits=5, smoothing=10):
    global_mean = tr_target.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
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

def clean_and_feature(df, is_train=True, kmeans_model=None):
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
    df['GroupID']      = df.index.to_series().str.split('_').str[0]
    df['FamilySize']   = df.groupby('GroupID')['GroupID'].transform('count')
    df['IsAlone']      = (df['FamilySize']==1).astype(int)
    df['Route']        = df['HomePlanet'] + "_" + df['Destination']
    df['Age']          = df['Age'].fillna(df['Age'].median())
    df['AgeBin']       = pd.cut(df['Age'], bins=[0,18,35,60,200],
                                labels=['child','young','adult','senior'])
    for c in spend_cols + ['Age']:
        df[f'Group_{c}_max'] = df.groupby('GroupID')[c].transform('max')
        df[f'Group_{c}_min'] = df.groupby('GroupID')[c].transform('min')
        df[f'Group_{c}_std'] = df.groupby('GroupID')[c].transform('std').fillna(0)
        df[f'Group_{c}_mean'] = df.groupby('GroupID')[c].transform('mean')
    df['Age_TotalSpend'] = df['Age'] * df['TotalSpend']
    df['CabinNum_even'] = (df['CabinNum'] % 2 == 0).astype(int)
    df['Spend_per_Age'] = df['TotalSpend'] / (df['Age'] + 1)
    df['VIP_CryoSleep'] = df['VIP'].astype(int) * df['CryoSleep'].astype(int)
    df['IsVIP_in_Group'] = (df.groupby('GroupID')['VIP'].transform('sum') > 0).astype(int)
    df['MissingCount'] = df.isnull().sum(axis=1)
    df['NameLength'] = df['Name'].str.len()
    numeric_cols = ['Age', 'TotalSpend'] + [c for c in df.columns if 'Group_' in c]
    if kmeans_model is None and is_train:
        kmeans_model = KMeans(n_clusters=8, random_state=RANDOM_STATE)
        kmeans_model.fit(df[numeric_cols].fillna(0))
    if kmeans_model is not None:
        df['Cluster'] = kmeans_model.predict(df[numeric_cols].fillna(0))
    df = df.drop(columns=['Name','Cabin'])
    if is_train:
        y = df['Transported'].astype(int)
        return df.drop(columns=['Transported']), y, kmeans_model
    else:
        return df, kmeans_model

print("Loading and preprocessing data...")
train_raw = pd.read_csv('train.csv', index_col='PassengerId')
test_raw  = pd.read_csv('test.csv',  index_col='PassengerId')
X_raw, y, kmeans_model = clean_and_feature(train_raw, is_train=True)
X_test_raw, _ = clean_and_feature(test_raw, is_train=False, kmeans_model=kmeans_model)

Xa = pd.concat([X_raw, X_test_raw], axis=0)
ya = np.concatenate([np.zeros(len(X_raw)), np.ones(len(X_test_raw))])
for col in Xa.select_dtypes(include='category').columns:
    Xa[col] = Xa[col].astype(str)
    X_raw[col] = X_raw[col].astype(str)
    X_test_raw[col] = X_test_raw[col].astype(str)
Xa = pd.get_dummies(Xa, drop_first=True)
adv = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
adv.fit(Xa.fillna(0), ya)
imp = pd.Series(adv.feature_importances_, index=Xa.columns).nlargest(5)
cols_to_drop = [col for col in imp.index if col in X_raw.columns]
X_raw = X_raw.drop(columns=cols_to_drop)
X_test_raw = X_test_raw.drop(columns=cols_to_drop)

numerics = [col for col in X_raw.select_dtypes(include=np.number).columns]
categoricals = [col for col in X_raw.select_dtypes(exclude=np.number).columns]
high_card = ['Title','Deck','Route']
low_card = [c for c in categoricals if c not in high_card]
high_card = [col for col in high_card if col in X_raw.columns]

ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
ohe.fit(X_raw[low_card])
X_ohe = pd.DataFrame(ohe.transform(X_raw[low_card]), index=X_raw.index,
                     columns=ohe.get_feature_names_out(low_card))
X_test_ohe = pd.DataFrame(ohe.transform(X_test_raw[low_card]), index=X_test_raw.index,
                          columns=ohe.get_feature_names_out(low_card))

# Only use numerics and one-hot encoded columns (no object columns)
X_base = pd.concat([X_raw[numerics], X_ohe], axis=1)
X_test_base = pd.concat([X_test_raw[numerics], X_test_ohe], axis=1)
groups = X_raw['GroupID']

scaler = StandardScaler()
X_base[numerics] = scaler.fit_transform(X_base[numerics])
X_test_base[numerics] = scaler.transform(X_test_base[numerics])

# Sanity check: all columns must be numeric
assert all([np.issubdtype(dtype, np.number) for dtype in X_base.dtypes]), "Non-numeric columns in X_base!"
assert all([np.issubdtype(dtype, np.number) for dtype in X_test_base.dtypes]), "Non-numeric columns in X_test_base!"

print("Optimizing LightGBM hyperparameters...")
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: optimize_lgb(trial, X_base, y), n_trials=100)
best_params = study.best_params

models = {
    'hgb': HistGradientBoostingClassifier(max_iter=1200, learning_rate=0.04, max_leaf_nodes=35, random_state=RANDOM_STATE),
    'lgb': LGBMClassifier(**best_params),
    'cat': CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, verbose=0, random_state=RANDOM_STATE),
    'xgb': XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
    'et' : ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE)
}

print("Training models...")
n_train = len(X_base)
n_test = len(X_test_base)
oof_preds = {k: np.zeros(n_train) for k in models}
test_preds = {k: np.zeros(n_test) for k in models}

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
for fold, (tr, va) in enumerate(sgkf.split(X_base, y, groups)):
    print(f"Training fold {fold + 1}/{N_SPLITS}")
    X_tr = X_base.iloc[tr].copy()
    X_va = X_base.iloc[va].copy()
    X_te = X_test_base.copy()
    for col in high_card:
        if col in X_tr.columns and col in X_va.columns and col in X_te.columns:
            tr_enc, va_enc = kfold_target_encode(X_tr[col], y.iloc[tr], X_va[col])
            _, te_enc = kfold_target_encode(X_tr[col], y.iloc[tr], X_te[col])
            X_tr[col] = tr_enc
            X_va[col] = va_enc
            X_te[col] = te_enc
    for name, mdl in models.items():
        mdl.fit(X_tr, y.iloc[tr])
        oof_preds[name][va] = mdl.predict_proba(X_va)[:,1]
        test_preds[name] += mdl.predict_proba(X_te)[:,1] / N_SPLITS

print("Performing SHAP analysis...")
explainer = shap.TreeExplainer(models['lgb'])
shap_values = explainer.shap_values(X_base)
shap.summary_plot(shap_values, X_base, show=False)
import matplotlib.pyplot as plt
plt.savefig('feature_importance.png')
plt.close()

print("Applying pseudo-labeling...")
meta_X = pd.DataFrame(oof_preds, index=X_base.index)
meta_model = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE)
meta_model.fit(meta_X, y)

test_pred = meta_model.predict_proba(pd.DataFrame(test_preds, index=X_test_base.index))[:,1]
confident_idx = (test_pred > PSEUDO_LABEL_THRESHOLD) | (test_pred < (1-PSEUDO_LABEL_THRESHOLD))
pseudo_X = X_test_base[confident_idx]
pseudo_y = (test_pred[confident_idx] > 0.5).astype(int)

X_aug = pd.concat([X_base, pseudo_X])
y_aug = pd.concat([y, pd.Series(pseudo_y, index=pseudo_X.index)])
meta_model.fit(pd.concat([meta_X, pd.DataFrame(test_preds, index=X_test_base.index)[confident_idx]]), y_aug)

print("Generating final predictions...")
test_df = pd.DataFrame(test_preds, index=X_test_base.index)
final_prob = meta_model.predict_proba(test_df)[:,1]

best_thr = 0.5
submission = pd.DataFrame({
    'PassengerId': X_test_base.index,
    'Transported': final_prob > best_thr
})
submission.to_csv('submission.csv', index=False)

print("✅ Done! Submission file created.")
print(f"Features used: {X_base.shape[1]}")
print(f"Pseudo-labeled samples: {sum(confident_idx)}")
