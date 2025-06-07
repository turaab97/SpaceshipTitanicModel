# tune_lgbm.py

import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 0) Feature‐engineering helper
def fe(df):
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['SpendBucket'] = pd.qcut(df['TotalSpend'], 4, labels=False, duplicates='drop')
    df['GroupID'] = df.index.to_series().str.split('_').str[0]
    df['FamilySize'] = df.groupby('GroupID')['GroupID'].transform('count')
    df['AgeBin'] = pd.cut(df['Age'], [0,18,35,60,200],
                         labels=['child','young','adult','senior'])
    return df

# 1) Load & featurize
df = pd.read_csv('train_cleaned.csv', index_col='PassengerId')
df = fe(df)

# 2) Prepare X/y
cat_cols = ['HomePlanet','Destination','Deck','Side','SpendBucket','AgeBin']
X = pd.get_dummies(
    df.drop(columns=['Transported','GroupID']),
    columns=cat_cols,
    drop_first=True
)
y = df['Transported'].astype(int)

# 3) CV + Optuna objective
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_idx, valid_idx in cv.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_va, label=y_va)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )

        preds = model.predict(X_va)
        aucs.append(roc_auc_score(y_va, preds))

    return sum(aucs) / len(aucs)

# 4) Run the study
if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print("Best mean AUC:", study.best_value)
    print("Best hyperparameters:", study.best_params)
