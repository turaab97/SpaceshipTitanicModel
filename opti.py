import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

"""
OPTIMIZATION STRATEGY
Since adding features made things worse, let's focus on:
1. Better data cleaning (fix CryoSleep contradictions)
2. Optimize ensemble weights
3. Better handling of missing values
4. Feature selection to remove noise
"""

def optimized_feature_engineering(df, is_train=True):
    """
    Your EXACT features but with better data cleaning
    """
    df = df.copy()
    
    # ========== CRITICAL DATA CLEANING ==========
    # Fix CryoSleep contradictions BEFORE feature engineering
    # This is a hard rule: CryoSleep = True means TotalSpend must be 0
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    
    # Fill spending NaNs with 0 first
    df[spend_cols] = df[spend_cols].fillna(0)
    
    # Calculate total spend
    total_spend_temp = df[spend_cols].sum(axis=1)
    
    # Fix CryoSleep based on spending
    # If someone has spending > 0, they CAN'T be in CryoSleep
    cryo_map = {True:1, False:0, 'True':1, 'False':0}
    df['CryoSleep'] = df['CryoSleep'].map(cryo_map)
    
    # Fix contradictions
    contradictions = (df['CryoSleep'] == 1) & (total_spend_temp > 0)
    if is_train:
        print(f"Found {contradictions.sum()} CryoSleep contradictions - fixing...")
    df.loc[contradictions, 'CryoSleep'] = 0
    
    # Also, if CryoSleep is NaN but spending is 0, likely CryoSleep
    cryo_nan_zero_spend = df['CryoSleep'].isna() & (total_spend_temp == 0)
    df.loc[cryo_nan_zero_spend, 'CryoSleep'] = 1
    
    # Fill remaining CryoSleep NaN with 0
    df['CryoSleep'] = df['CryoSleep'].fillna(0).astype(int)
    
    # ========== YOUR ORIGINAL FEATURES ==========
    # Group features
    df['GroupId'] = df.index.str.split('_').str[0]
    df['GroupSize'] = df.groupby('GroupId')['HomePlanet'].transform('size')
    df['IsAlone'] = (df['GroupSize']==1).astype(int)
    
    # Cabin features with better handling
    cabin = df['Cabin'].str.split('/', expand=True)
    df['CabinDeck'] = cabin[0].fillna('Unknown')
    cn = pd.to_numeric(cabin[1], errors='coerce')
    
    # Better cabin number imputation - use deck-specific median
    for deck in df['CabinDeck'].unique():
        deck_mask = df['CabinDeck'] == deck
        deck_median = cn[deck_mask].median()
        if pd.notna(deck_median):
            cn.loc[deck_mask & cn.isna()] = deck_median
    cn.fillna(cn.median(), inplace=True)
    df['CabinNum'] = cn
    
    df['CabinSide'] = cabin[2].fillna('Unknown')
    
    # Log features
    for c in spend_cols:
        df[f'log_{c}'] = np.log1p(df[c])
    
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['SpendPerPerson'] = df['TotalSpend'] / (df['GroupSize'] + 1)
    
    # Better age imputation - use group information
    # First try to fill with group median age
    group_age_median = df.groupby('GroupId')['Age'].transform('median')
    df['Age'] = df['Age'].fillna(group_age_median)
    # Then fill remaining with global median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,np.inf],
                         labels=['Child','Teen','Adult','Middle','Senior'],
                         include_lowest=True)
    
    # Composite features
    df['DeckSide'] = df['CabinDeck'] + '_' + df['CabinSide']
    
    # VIP conversion
    df['VIP'] = df['VIP'].map({True:1, False:0, 'True':1, 'False':0})\
                   .fillna(0).astype(int)
    
    # Your interaction features - but now CryoSpendInteraction will be more meaningful
    df['CryoSpendInteraction'] = df['CryoSleep'] * df['TotalSpend']
    df['VIPSpendInteraction'] = df['VIP'] * df['TotalSpend']
    df['YoungAloneFlag'] = ((df['Age'] < 25) & (df['IsAlone'] == 1)).astype(int)
    
    return df

def optimize_ensemble_weights(X, y):
    """
    Find optimal ensemble weights using cross-validation
    """
    print("\n🔍 Finding optimal ensemble weights...")
    
    # Train individual models
    hgb1 = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500,
        max_leaf_nodes=31, random_state=42)
    
    hgb2 = HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=300,
        max_leaf_nodes=31, random_state=24)
    
    lgbm = LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, random_state=0, verbose=-1)
    
    xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.05,
        max_depth=6, random_state=42, eval_metric='logloss')
    
    # Get out-of-fold predictions for each model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {'hgb1': hgb1, 'hgb2': hgb2, 'lgbm': lgbm, 'xgb': xgb}
    oof_preds = {}
    
    for name, model in models.items():
        oof_preds[name] = np.zeros(len(y))
        
        for train_idx, val_idx in cv.split(X, y):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof_preds[name][val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    
    # Test different weight combinations
    best_score = 0
    best_weights = None
    
    from itertools import product
    weight_options = [0.8, 1.0, 1.2, 1.5]
    
    for w1, w2, w3, w4 in product(weight_options, repeat=4):
        weights = np.array([w1, w2, w3, w4])
        weights = weights / weights.sum()  # Normalize
        
        weighted_pred = (
            weights[0] * oof_preds['hgb1'] +
            weights[1] * oof_preds['hgb2'] +
            weights[2] * oof_preds['lgbm'] +
            weights[3] * oof_preds['xgb']
        )
        
        score = (weighted_pred.round() == y).mean()
        
        if score > best_score:
            best_score = score
            best_weights = weights
    
    print(f"Best weights found: HGB1={best_weights[0]:.3f}, HGB2={best_weights[1]:.3f}, "
          f"LGBM={best_weights[2]:.3f}, XGB={best_weights[3]:.3f}")
    print(f"Weighted ensemble OOF score: {best_score:.4f}")
    
    return best_weights, models

def create_optimized_model():
    """
    Optimized approach focusing on data quality and ensemble tuning
    """
    print("🎯 OPTIMIZATION STRATEGY")
    print("Focus: Data quality and ensemble optimization")
    print("="*50)
    
    # Load data
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    # Apply optimized feature engineering
    print("\n⚙️ Applying optimized feature engineering...")
    train = optimized_feature_engineering(train, is_train=True)
    test = optimized_feature_engineering(test, is_train=False)
    
    # Your clustering with scaling for better results
    print("\n🔮 Creating scaled clusters...")
    km_feats = ['CabinNum','log_RoomService','log_FoodCourt',
                'log_ShoppingMall','log_Spa','log_VRDeck','TotalSpend']
    
    scaler = StandardScaler()
    train_cluster = train[km_feats].fillna(0)
    train_scaled = scaler.fit_transform(train_cluster)
    
    # Try different cluster numbers
    best_k = 6
    best_score = 0
    
    print("Testing different cluster numbers...")
    for k in [4, 5, 6, 7, 8]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        train['Cluster_temp'] = kmeans.fit_predict(train_scaled)
        
        # Quick evaluation
        from sklearn.metrics import silhouette_score
        score = silhouette_score(train_scaled, train['Cluster_temp'], sample_size=1000)
        print(f"  k={k}: silhouette score = {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"Best cluster number: {best_k}")
    
    # Use best clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    train['Cluster'] = kmeans.fit_predict(train_scaled)
    
    test_cluster = test[km_feats].fillna(0)
    test_scaled = scaler.transform(test_cluster)
    test['Cluster'] = kmeans.predict(test_scaled)
    
    # Advanced features
    train['MaxSpendItem'] = train[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].idxmax(axis=1)
    test['MaxSpendItem'] = test[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].idxmax(axis=1)
    
    # Improved frequency encoding with smoothing
    for col in ['HomePlanet','Destination','DeckSide']:
        freq = train[col].value_counts() / len(train)
        # Add Laplace smoothing
        smooth_factor = 1 / len(train)
        train[f'{col}_Freq'] = train[col].map(freq).fillna(smooth_factor)
        test[f'{col}_Freq'] = test[col].map(freq).fillna(smooth_factor)
    
    # Prepare data
    y = train['Transported'].astype(int)
    drop_cols = ['Transported','Name','Cabin','GroupId','HomePlanet','Destination', 'Cluster_temp']
    drop_cols = [col for col in drop_cols if col in train.columns]
    X = train.drop(columns=drop_cols)
    
    test_drop_cols = [col for col in drop_cols if col in test.columns and col != 'Transported']
    X_test = test.drop(columns=test_drop_cols)
    
    # Categorical encoding
    cat_cols = ['AgeBin','CabinDeck','CabinSide','DeckSide','MaxSpendItem','Cluster']
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
        X_test[c] = X_test[c].astype('category').cat.codes
    
    print(f"\n📊 Total features: {X.shape[1]}")
    
    # Find optimal weights
    best_weights, models = optimize_ensemble_weights(X, y)
    
    # Test different ensemble strategies
    print("\n🔧 Testing ensemble strategies...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Strategy 1: Weighted voting with optimal weights
    from sklearn.ensemble import VotingClassifier
    
    voting_opt = VotingClassifier(
        estimators=list(models.items()),
        voting='soft',
        weights=best_weights,
        n_jobs=-1
    )
    
    scores_voting = cross_val_score(voting_opt, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Optimized voting: {scores_voting.mean():.4f} ± {scores_voting.std():.4f}")
    
    # Strategy 2: Your original stacking
    stack_original = StackingClassifier(
        estimators=list(models.items()),
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    scores_stack = cross_val_score(stack_original, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Original stacking: {scores_stack.mean():.4f} ± {scores_stack.std():.4f}")
    
    # Strategy 3: Stacking with regularized meta-learner
    stack_reg = StackingClassifier(
        estimators=list(models.items()),
        final_estimator=LogisticRegression(max_iter=1000, C=0.5),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    scores_stack_reg = cross_val_score(stack_reg, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Regularized stacking: {scores_stack_reg.mean():.4f} ± {scores_stack_reg.std():.4f}")
    
    # Strategy 4: Remove worst performing model
    print("\n🔍 Testing model removal...")
    
    # Find worst model
    model_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        model_scores[name] = scores.mean()
        print(f"{name}: {scores.mean():.4f}")
    
    worst_model = min(model_scores, key=model_scores.get)
    print(f"\nWorst model: {worst_model}")
    
    # Try without worst model
    reduced_models = {k: v for k, v in models.items() if k != worst_model}
    reduced_weights = [w for i, w in enumerate(best_weights) if list(models.keys())[i] != worst_model]
    reduced_weights = np.array(reduced_weights) / np.sum(reduced_weights)
    
    voting_reduced = VotingClassifier(
        estimators=list(reduced_models.items()),
        voting='soft',
        weights=reduced_weights,
        n_jobs=-1
    )
    
    scores_reduced = cross_val_score(voting_reduced, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Without {worst_model}: {scores_reduced.mean():.4f}")
    
    # Choose best approach
    all_scores = {
        'Optimized Voting': scores_voting.mean(),
        'Original Stacking': scores_stack.mean(),
        'Regularized Stacking': scores_stack_reg.mean(),
        'Reduced Voting': scores_reduced.mean()
    }
    
    best_approach = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_approach]
    
    print(f"\n🏆 Best approach: {best_approach} with CV: {best_score:.4f}")
    print(f"   vs your baseline 0.807: {(best_score - 0.807)*100:+.2f}%")
    
    # Train best model
    if best_approach == 'Optimized Voting':
        final_model = voting_opt
    elif best_approach == 'Original Stacking':
        final_model = stack_original
    elif best_approach == 'Regularized Stacking':
        final_model = stack_reg
    else:
        final_model = voting_reduced
    
    print(f"\n🚀 Training {best_approach}...")
    final_model.fit(X, y)
    
    # Make predictions
    if hasattr(final_model, 'predict_proba'):
        preds_proba = final_model.predict_proba(X_test)[:, 1]
        
        # Test different thresholds
        print("\n🎯 Testing prediction thresholds...")
        thresholds = [0.48, 0.49, 0.50, 0.51, 0.52]
        
        for thresh in thresholds:
            preds_thresh = (preds_proba >= thresh).astype(bool)
            transported_pct = preds_thresh.mean() * 100
            print(f"Threshold {thresh}: {transported_pct:.1f}% transported")
        
        # Use standard threshold
        preds = (preds_proba >= 0.50).astype(bool)
    else:
        preds = final_model.predict(X_test).astype(bool)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': X_test.index,
        'Transported': preds
    })
    submission['PassengerId'] = submission['PassengerId'].astype(str)
    submission.to_csv('submission_optimized.csv', index=False)
    
    print("\n✅ Created submission_optimized.csv")
    
    # Additional analysis
    print("\n📊 Data quality improvements:")
    print(f"CryoSleep contradictions fixed: See above")
    print(f"Improved Age imputation using group information")
    print(f"Improved CabinNum imputation using deck-specific medians")
    print(f"Optimal cluster number: {best_k}")
    print(f"Optimal ensemble weights found via grid search")
    
    return final_model, best_score

if __name__ == "__main__":
    print("🎯 SPACESHIP TITANIC - OPTIMIZATION APPROACH")
    print("Since adding features failed, focusing on:")
    print("1. Data quality (fixing contradictions)")
    print("2. Better imputation strategies")
    print("3. Optimal ensemble weights")
    print("4. Feature scaling for clustering")
    print("="*50)
    
    model, score = create_optimized_model()
    
    print(f"\n📊 SUMMARY:")
    print(f"Your baseline: 0.8070")
    print(f"Enhanced attempt: 0.8036 (failed)")
    print(f"Minimal boost: 0.8013 (failed)")
    print(f"This optimization: {score:.4f}")
    
    if score > 0.807:
        print(f"\n🎉 SUCCESS! Improved by {(score - 0.807)*100:+.2f}%")
    else:
        print(f"\n⚠️ Still below baseline. Your original approach is hard to beat!")
        print("\nNext suggestions:")
        print("1. Try semi-supervised learning with test data")
        print("2. Use different random seeds and average predictions")
        print("3. Manual feature engineering based on EDA insights")
