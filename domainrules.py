import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

"""
POST-PROCESSING STRATEGY
1. Use your exact 0.807 baseline
2. Add diverse models for better ensemble
3. Apply domain-specific post-processing rules
4. Use multiple random seeds and average
"""

def your_exact_feature_engineering(df):
    """
    Your EXACT feature engineering that got 0.807
    """
    df = df.copy()
    
    # Group features
    df['GroupId'] = df.index.str.split('_').str[0]
    df['GroupSize'] = df.groupby('GroupId')['HomePlanet'].transform('size')
    df['IsAlone'] = (df['GroupSize']==1).astype(int)
    
    # Cabin features
    cabin = df['Cabin'].str.split('/', expand=True)
    df['CabinDeck'] = cabin[0].fillna('Unknown')
    cn = pd.to_numeric(cabin[1], errors='coerce')
    df['CabinNum'] = cn.fillna(cn.median())
    df['CabinSide'] = cabin[2].fillna('Unknown')
    
    # Spending features
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    df[spend_cols] = df[spend_cols].fillna(0)
    
    # Log features
    for c in spend_cols:
        df[f'log_{c}'] = np.log1p(df[c])
    
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['SpendPerPerson'] = df['TotalSpend'] / (df['GroupSize'] + 1)
    
    # Age features
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,np.inf],
                         labels=['Child','Teen','Adult','Middle','Senior'],
                         include_lowest=True)
    
    # Composite features
    df['DeckSide'] = df['CabinDeck'] + '_' + df['CabinSide']
    
    # Boolean conversion
    for col in ['CryoSleep','VIP']:
        df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0})\
                       .fillna(0).astype(int)
    
    # Your interaction features
    df['CryoSpendInteraction'] = df['CryoSleep'] * df['TotalSpend']
    df['VIPSpendInteraction'] = df['VIP'] * df['TotalSpend']
    df['YoungAloneFlag'] = ((df['Age'] < 25) & (df['IsAlone'] == 1)).astype(int)
    
    return df

def apply_post_processing_rules(df, predictions, proba=None):
    """
    Apply domain-specific rules to correct predictions
    """
    # Convert to pandas Series if numpy array
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions, index=df.index)
    else:
        predictions = predictions.copy()
    
    corrections = 0
    
    # Rule 1: Strong CryoSleep rule
    # If CryoSleep=1 and TotalSpend=0, very likely transported
    cryo_no_spend = (df['CryoSleep'] == 1) & (df['TotalSpend'] == 0)
    if proba is not None:
        # Only override if model is uncertain (0.3 < proba < 0.7)
        uncertain = (proba > 0.3) & (proba < 0.7)
        override_mask = cryo_no_spend & uncertain
        corrections += override_mask.sum()
        predictions[override_mask] = True
    
    # Rule 2: Group consistency
    # Small groups (2-3 people) often have same outcome
    if 'GroupId' in df.columns:
        small_groups = df[df['GroupSize'].between(2, 3)]['GroupId'].unique()
        
        for group in small_groups:
            group_mask = df['GroupId'] == group
            group_indices = df[group_mask].index
            group_preds = predictions[group_indices]
            
            # If group has strong majority, apply to uncertain members
            if len(group_preds) >= 2:
                transported_ratio = group_preds.mean()
                
                if transported_ratio >= 0.75:  # Strong majority transported
                    if proba is not None:
                        uncertain_mask = (proba[group_indices] > 0.4) & (proba[group_indices] < 0.6)
                        uncertain_indices = group_indices[uncertain_mask]
                        corrections += (~predictions[uncertain_indices]).sum()
                        predictions[uncertain_indices] = True
                        
                elif transported_ratio <= 0.25:  # Strong majority not transported
                    if proba is not None:
                        uncertain_mask = (proba[group_indices] > 0.4) & (proba[group_indices] < 0.6)
                        uncertain_indices = group_indices[uncertain_mask]
                        corrections += predictions[uncertain_indices].sum()
                        predictions[uncertain_indices] = False
    
    # Rule 3: Age-based patterns
    # Very young children (< 5) with parents likely have same outcome
    young_children = df['Age'] < 5
    if young_children.sum() > 0 and 'GroupSize' in df.columns:
        young_indices = df[young_children].index
        
        for idx in young_indices:
            if df.loc[idx, 'GroupSize'] > 1:  # Not alone
                group_id = df.loc[idx, 'GroupId']
                adult_mask = (df['GroupId'] == group_id) & (df['Age'] > 18)  # Adults in group
                adult_indices = df[adult_mask].index
                
                if len(adult_indices) > 0:
                    adult_pred = predictions[adult_indices].mean()
                    if adult_pred > 0.7 and not predictions[idx]:
                        predictions[idx] = True
                        corrections += 1
                    elif adult_pred < 0.3 and predictions[idx]:
                        predictions[idx] = False
                        corrections += 1
    
    print(f"Post-processing corrections: {corrections}")
    
    # Convert back to numpy array if needed
    return predictions.values if isinstance(predictions, pd.Series) else predictions

def create_diverse_ensemble():
    """
    Create ensemble with diverse models and post-processing
    """
    print("🎯 POST-PROCESSING STRATEGY")
    print("Using diverse models and domain rules")
    print("="*50)
    
    # Load data
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    # Keep original data for post-processing
    train_original = train.copy()
    test_original = test.copy()
    
    # Apply your exact feature engineering
    print("⚙️ Applying your exact feature engineering...")
    train = your_exact_feature_engineering(train)
    test = your_exact_feature_engineering(test)
    
    # Your exact clustering
    km_feats = train[['CabinNum','log_RoomService','log_FoodCourt',
                      'log_ShoppingMall','log_Spa','log_VRDeck','TotalSpend']].fillna(0)
    kmeans = KMeans(n_clusters=6, random_state=42).fit(km_feats)
    train['Cluster'] = kmeans.predict(km_feats)
    test['Cluster'] = kmeans.predict(test[km_feats.columns].fillna(0))
    
    # Your exact advanced features
    train['MaxSpendItem'] = train[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].idxmax(axis=1)
    test['MaxSpendItem'] = test[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].idxmax(axis=1)
    
    for col in ['HomePlanet','Destination','DeckSide']:
        freq = train[col].value_counts() / len(train)
        train[f'{col}_Freq'] = train[col].map(freq).fillna(0)
        test[f'{col}_Freq'] = test[col].map(freq).fillna(0)
    
    # Prepare data
    y = train['Transported'].astype(int)
    drop_cols = ['Transported','Name','Cabin','GroupId','HomePlanet','Destination']
    X = train.drop(columns=drop_cols)
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])
    
    # Keep GroupId for post-processing
    test['GroupId'] = test_original.index.str.split('_').str[0]
    
    # Categorical encoding
    cat_cols = ['AgeBin','CabinDeck','CabinSide','DeckSide','MaxSpendItem','Cluster']
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
        X_test[c] = X_test[c].astype('category').cat.codes
    
    print(f"📊 Features: {X.shape[1]}")
    
    # Create DIVERSE models
    print("\n🔧 Creating diverse ensemble...")
    
    # Your original models
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
    
    # Add diverse models
    catboost = CatBoostClassifier(
        iterations=500, learning_rate=0.05,
        depth=6, random_seed=42, verbose=False)
    
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10,
        min_samples_split=20, random_state=42, n_jobs=-1)
    
    # Test different ensemble combinations
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n📊 Testing ensemble combinations...")
    
    # Original ensemble
    ensemble1 = [('hgb1', hgb1), ('hgb2', hgb2), ('lgbm', lgbm), ('xgb', xgb)]
    
    # Add CatBoost
    ensemble2 = ensemble1 + [('catboost', catboost)]
    
    # Add Random Forest for diversity
    ensemble3 = ensemble2 + [('rf', rf)]
    
    # Test each ensemble
    ensembles = {
        'Original (4 models)': ensemble1,
        'With CatBoost (5 models)': ensemble2,
        'With CatBoost + RF (6 models)': ensemble3
    }
    
    best_ensemble_name = None
    best_ensemble_score = 0
    best_ensemble = None
    
    for name, ensemble in ensembles.items():
        stack = StackingClassifier(
            estimators=ensemble,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5,
            passthrough=True,
            n_jobs=-1
        )
        
        scores = cross_val_score(stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
        
        if scores.mean() > best_ensemble_score:
            best_ensemble_score = scores.mean()
            best_ensemble_name = name
            best_ensemble = ensemble
    
    print(f"\n🏆 Best ensemble: {best_ensemble_name}")
    
    # Multi-seed averaging
    print("\n🎲 Training with multiple seeds for stability...")
    
    seeds = [42, 24, 0, 123, 999]
    all_predictions = []
    all_probas = []
    
    for i, seed in enumerate(seeds):
        print(f"Training with seed {seed}...")
        
        # Update random states
        for model_name, model in best_ensemble:
            if hasattr(model, 'random_state'):
                model.random_state = seed
            elif hasattr(model, 'random_seed'):
                model.random_seed = seed
        
        # Create stacking with updated models
        stack = StackingClassifier(
            estimators=best_ensemble,
            final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
            cv=5,
            passthrough=True,
            n_jobs=-1
        )
        
        # Train and predict
        stack.fit(X, y)
        pred_proba = stack.predict_proba(X_test)[:, 1]
        pred = stack.predict(X_test)
        
        all_probas.append(pred_proba)
        all_predictions.append(pred)
    
    # Average probabilities
    avg_proba = np.mean(all_probas, axis=0)
    avg_pred = (avg_proba >= 0.5).astype(bool)
    
    print(f"\n📊 Prediction statistics:")
    print(f"Average probability range: [{avg_proba.min():.3f}, {avg_proba.max():.3f}]")
    print(f"Uncertain predictions (0.4-0.6): {((avg_proba > 0.4) & (avg_proba < 0.6)).sum()}")
    
    # Apply post-processing
    print("\n🔧 Applying post-processing rules...")
    
    # Prepare test data for post-processing
    test_for_rules = test.copy()
    test_for_rules['GroupId'] = test['GroupId']
    test_for_rules['GroupSize'] = test['GroupSize']
    test_for_rules['CryoSleep'] = test['CryoSleep']
    test_for_rules['TotalSpend'] = test['TotalSpend']
    test_for_rules['Age'] = test['Age']
    
    # Convert proba to pandas Series with proper index
    avg_proba_series = pd.Series(avg_proba, index=X_test.index)
    
    # Apply rules
    final_predictions = apply_post_processing_rules(
        test_for_rules, avg_pred, avg_proba_series
    )
    
    # Create submissions
    submission_avg = pd.DataFrame({
        'PassengerId': X_test.index,
        'Transported': avg_pred
    })
    submission_avg['PassengerId'] = submission_avg['PassengerId'].astype(str)
    submission_avg.to_csv('submission_averaged.csv', index=False)
    
    submission_post = pd.DataFrame({
        'PassengerId': X_test.index,
        'Transported': final_predictions
    })
    submission_post['PassengerId'] = submission_post['PassengerId'].astype(str)
    submission_post.to_csv('submission_postprocessed.csv', index=False)
    
    print("\n✅ Created two submissions:")
    print("1. submission_averaged.csv - Multi-seed averaging")
    print("2. submission_postprocessed.csv - With domain rules")
    
    # Analysis
    print("\n📊 Prediction differences:")
    differences = (avg_pred != final_predictions).sum()
    print(f"Post-processing changed {differences} predictions ({differences/len(avg_pred)*100:.1f}%)")
    
    return stack, best_ensemble_score

if __name__ == "__main__":
    print("🎯 SPACESHIP TITANIC - POST-PROCESSING STRATEGY")
    print("New approach: Ensemble diversity + Domain rules")
    print("="*50)
    
    model, score = create_diverse_ensemble()
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"Your baseline: 0.8070")
    print(f"Previous attempts: 0.8036, 0.8013, 0.8066 (all worse)")
    print(f"This approach CV: {score:.4f}")
    
    print("\n💡 Key strategies used:")
    print("1. Added CatBoost and RandomForest for diversity")
    print("2. Multi-seed averaging (5 seeds) for stability")
    print("3. Post-processing with domain rules:")
    print("   - CryoSleep + no spending = transported")
    print("   - Small group consistency")
    print("   - Parent-child outcome matching")
    print("\n🎯 Try both submissions - one might break through!")
