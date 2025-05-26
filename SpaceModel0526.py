# Import required libraries
import pandas as pd
import numpy as np
from scipy.stats import randint, uniform

# Import cross-validation splitters
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV

# Import preprocessing tools
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

# Import models
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Import metrics
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from itertools import product

# Set pandas options
pd.set_option('future.no_silent_downcasting', True)

# Define spending columns
spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

def kfold_target_encode(tr_series, tr_target, val_series,
                       n_splits=5, smoothing=10, random_state=42):
    """
    Perform target encoding with K-Fold to avoid leakage.
    """
    global_mean = tr_target.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = pd.Series(np.nan, index=tr_series.index)

    for tr_idx, val_idx in kf.split(tr_series):
        fold_tr = tr_series.iloc[tr_idx]
        fold_val = tr_series.iloc[val_idx]
        fold_target = tr_target.iloc[tr_idx]
        
        df = pd.DataFrame({'cat': fold_tr, 'y': fold_target})
        agg = df.groupby('cat')['y'].agg(['mean', 'count'])
        agg['smooth'] = (
            (agg['mean'] * agg['count'] + global_mean * smoothing)
            / (agg['count'] + smoothing)
        )
        oof.iloc[val_idx] = fold_val.map(agg['smooth']).fillna(global_mean)

    df_full = pd.DataFrame({'cat': tr_series, 'y': tr_target})
    agg_full = df_full.groupby('cat')['y'].agg(['mean', 'count'])
    agg_full['smooth'] = (
        (agg_full['mean'] * agg_full['count'] + global_mean * smoothing)
        / (agg_full['count'] + smoothing)
    )
    val_encoded = val_series.map(agg_full['smooth']).fillna(global_mean)

    return oof, val_encoded

def clean_and_feature(df, is_train=True):
    """
    Clean raw DataFrame and create features.
    """
    df = df.copy()

    # Basic cleaning
    df['HomePlanet'] = df['HomePlanet'].fillna('Unknown')
    df['Destination'] = df['Destination'].fillna('Unknown')
    df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(bool)
    df['VIP'] = df['VIP'].fillna(False).astype(bool)
    
    # Extract title and cabin info
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.').fillna('Unknown')
    df['Cabin'] = df['Cabin'].fillna('Unknown/0/Unknown')
    cab = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cab[0]
    df['CabinNum'] = pd.to_numeric(cab[1], errors='coerce').fillna(0).astype(int)
    df['Side'] = cab[2]

    # Handle spending columns
    for c in spend_cols:
        df[c] = df[c].fillna(0.0)
        df[f'{c}_was_missing'] = df[c].isna().astype(int)

    # Basic features
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    for c in spend_cols:
        df[f'{c}_ratio'] = df[c] / (df['TotalSpend'] + 1e-9)

    # Advanced features
    df['HasSpent'] = (df['TotalSpend'] > 0).astype(int)
    df['SpendingServices'] = df[spend_cols].apply(lambda x: (x > 0).sum(), axis=1)
    df['MaxSpendService'] = df[spend_cols].apply(lambda x: x.max())
    df['SpendVariance'] = df[spend_cols].apply(lambda x: x.var())

    # Spending patterns
    spend_scaled = StandardScaler().fit_transform(df[spend_cols])
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['SpendingCluster'] = kmeans.fit_predict(spend_scaled)

    # Group and family features
    df['GroupID'] = df.index.to_series().str.split('_').str[0]
    df['FamilySize'] = df.groupby('GroupID')['GroupID'].transform('count')
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Age features
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Age_squared'] = df['Age'] ** 2
    df['Age_decade'] = (df['Age'] // 10) * 10
    df['IsMinor'] = (df['Age'] < 18).astype(int)
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 200],
                         labels=['child', 'young', 'adult', 'senior'])

    # Cabin features
    df['CabinSection'] = df['CabinNum'] // 100
    df['CabinPosition'] = df['CabinNum'] % 100
    df['IsCabinEdge'] = ((df['CabinPosition'] <= 5) | 
                         (df['CabinPosition'] >= 95)).astype(int)

    # Route features
    df['Route'] = df['HomePlanet'] + "_" + df['Destination']
    route_volume = df['Route'].value_counts()
    df['RouteVolume'] = df['Route'].map(route_volume)

    # Group-level statistics
    for c in spend_cols + ['Age']:
        grp = df.groupby('GroupID')[c]
        df[f'Group_{c}_max'] = grp.transform('max')
        df[f'Group_{c}_min'] = grp.transform('min')
        df[f'Group_{c}_std'] = grp.transform('std').fillna(0)
        df[f'Group_{c}_mean'] = grp.transform('mean')
        df[f'{c}_group_rank'] = grp.rank()
        df[f'{c}_group_mean_diff'] = df[c] - df[f'Group_{c}_mean']

    # Interaction features
    df['Age_TotalSpend'] = df['Age'] * df['TotalSpend']
    df['VIP_TotalSpend'] = df['VIP'].astype(int) * df['TotalSpend']
    df['VIP_FamilySize'] = df['VIP'].astype(int) * df['FamilySize']
    df['CabinNum_even'] = (df['CabinNum'] % 2 == 0).astype(int)
    df['Spend_per_Age'] = df['TotalSpend'] / (df['Age'] + 1)
    df['VIP_CryoSleep'] = df['VIP'].astype(int) * df['CryoSleep'].astype(int)

    # Drop raw columns
    df = df.drop(columns=['Name', 'Cabin'])

    if is_train:
        y = df['Transported'].astype(int)
        X = df.drop(columns=['Transported'])
        return X, y
    else:
        return df

def main():
    # Load data
    train_raw = pd.read_csv('train.csv', index_col='PassengerId')
    test_raw = pd.read_csv('test.csv', index_col='PassengerId')

    # Clean and feature engineer
    X_raw, y = clean_and_feature(train_raw, is_train=True)
    X_test_raw = clean_and_feature(test_raw, is_train=False)

    # Prepare for encoding
    numerics = [col for col in X_raw.columns if X_raw[col].dtype in ['int64', 'float64']]
    categoricals = [col for col in X_raw.columns if col not in numerics]
    high_card = ['Title', 'Deck', 'Route']
    low_card = [c for c in categoricals if c not in high_card]

    # One-hot encode low cardinality categoricals
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    ohe.fit(X_raw[low_card])

    X_ohe = pd.DataFrame(
        ohe.transform(X_raw[low_card]),
        index=X_raw.index,
        columns=ohe.get_feature_names_out(low_card)
    )
    X_test_ohe = pd.DataFrame(
        ohe.transform(X_test_raw[low_card]),
        index=X_test_raw.index,
        columns=ohe.get_feature_names_out(low_card)
    )

    # Combine features
   # Combine features (exclude high_card columns for model tuning)
    X_base = pd.concat([X_raw[numerics], X_ohe], axis=1)
    X_test_base = pd.concat([X_test_raw[numerics], X_test_ohe], axis=1)
    groups = X_raw['GroupID']

    # Scale numeric features
    scaler = StandardScaler()
    X_base[numerics] = scaler.fit_transform(X_base[numerics])
    X_test_base[numerics] = scaler.transform(X_test_base[numerics])

    # Define parameter distributions for tuning
    param_distributions = {
        'hgb': {
            'max_iter': randint(800, 1500),
            'learning_rate': uniform(0.01, 0.1),
            'max_leaf_nodes': randint(20, 50),
            'min_samples_leaf': randint(10, 30),
            'max_depth': randint(3, 10),
            'l2_regularization': uniform(0, 2)
        },
        'mlp': {
            'hidden_layer_sizes': [(n,) for n in range(80, 150, 10)],
            'learning_rate_init': uniform(0.0001, 0.01),
            'alpha': uniform(0.0001, 0.01),
            'batch_size': randint(32, 256)
        },
        'lgb': {
            'n_estimators': randint(400, 1000),
            'learning_rate': uniform(0.01, 0.1),
            'num_leaves': randint(20, 100),
            'max_depth': randint(3, 10),
            'min_child_samples': randint(10, 50),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4)
        },
        'cat': {
            'iterations': randint(300, 700),
            'learning_rate': uniform(0.01, 0.1),
            'depth': randint(4, 10),
            'l2_leaf_reg': uniform(1, 10),
            'border_count': randint(32, 255)
        },
        'xgb': {
            'n_estimators': randint(300, 700),
            'learning_rate': uniform(0.01, 0.1),
            'max_depth': randint(3, 10),
            'min_child_weight': randint(1, 7),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4)
        },
        'et': {
            'n_estimators': randint(200, 500),
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
    }

    # Initialize base models
    base_models = {
        'hgb': HistGradientBoostingClassifier(random_state=42),
        'mlp': MLPClassifier(max_iter=1000, random_state=42),
        'lgb': LGBMClassifier(random_state=42),
        'cat': CatBoostClassifier(random_state=42, verbose=0),
        'xgb': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'et': ExtraTreesClassifier(random_state=42)
    }

    # Tune models
    print("Starting model tuning...")
    tuned_models = {}
    for name, model in base_models.items():
        print(f"\nTuning {name}...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[name],
            n_iter=25,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        search.fit(X_base, y)
        tuned_models[name] = search.best_estimator_
        print(f"Best score for {name}: {search.best_score_:.4f}")

    # Prepare for stacking
    n_train = len(X_base)
    n_test = len(X_test_base)
    oof_preds = {k: np.zeros(n_train) for k in tuned_models}
    test_preds = {k: np.zeros(n_test) for k in tuned_models}

    # Train models with cross-validation
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(sgkf.split(X_base, y, groups), 1):
        print(f"\nFold {fold}")
        X_tr = X_base.iloc[tr].copy()
        X_va = X_base.iloc[va].copy()
        X_te = X_test_base.copy()

        # Target encode high cardinality features
        for col in high_card:
            tr_enc, va_enc = kfold_target_encode(X_tr[col], y.iloc[tr], X_va[col])
            _, te_enc = kfold_target_encode(X_tr[col], y.iloc[tr], X_te[col])
            X_tr[col] = tr_enc
            X_va[col] = va_enc
            X_te[col] = te_enc

        # Train and predict with each model
        for name, model in tuned_models.items():
            print(f"Training {name}...")
            model.fit(X_tr, y.iloc[tr])
            oof_preds[name][va] = model.predict_proba(X_va)[:,1]
            test_preds[name] += model.predict_proba(X_te)[:,1] / 5

    # Create meta-features
    meta_X = pd.DataFrame(oof_preds, index=X_base.index)
    meta_test = pd.DataFrame(test_preds, index=X_test_base.index)

    # Train meta-model
    meta_model = LGBMClassifier(random_state=42)
    meta_model.fit(meta_X, y)

    # Make final predictions
    final_prob = meta_model.predict_proba(meta_test)[:,1]

    # Find optimal threshold
    thresholds = np.linspace(0.3, 0.7, 41)
    best_threshold = 0.5
    best_f1 = 0
    meta_oof_prob = meta_model.predict_proba(meta_X)[:,1]
    
    for threshold in thresholds:
        f1 = f1_score(y, (meta_oof_prob > threshold).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Create submission
    submission = pd.DataFrame({
        'PassengerId': X_test_base.index,
        'Transported': final_prob > best_threshold
    })
    submission.to_csv('submission.csv', index=False)

    # Print final scores
    print("\nFinal Scores:")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"OOF ROC-AUC: {roc_auc_score(y, meta_oof_prob):.4f}")
    print(f"OOF F1: {f1_score(y, (meta_oof_prob > best_threshold).astype(int)):.4f}")

if __name__ == "__main__":
    main()
