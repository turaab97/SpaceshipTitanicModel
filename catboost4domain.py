import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

"""
FINAL PUSH - PRECISION TUNING
Goal: Beat 80.8% with careful, targeted improvements

Key insights from your results:
- Original approach: 80.804% ✅
- Targeted approach: 80.757% (very close!)
- Over-engineered: 78.840% ❌

Strategy: Take targeted approach and make micro-improvements
"""

def enhanced_feature_engineering(df):
    """
    Your original features + carefully selected enhancements
    """
    df = df.copy()
    
    # YOUR EXACT ORIGINAL FEATURES
    df['GroupId'] = df.index.str.split('_').str[0]
    df['GroupSize'] = df.groupby('GroupId')['HomePlanet'].transform('size')
    df['IsAlone'] = (df['GroupSize']==1).astype(int)
    
    cabin = df['Cabin'].str.split('/', expand=True)
    df['CabinDeck'] = cabin[0].fillna('Unknown')
    cn = pd.to_numeric(cabin[1], errors='coerce')
    df['CabinNum'] = cn.fillna(cn.median())
    df['CabinSide'] = cabin[2].fillna('Unknown')
    
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    df[spend_cols] = df[spend_cols].fillna(0)
    
    for c in spend_cols:
        df[f'log_{c}'] = np.log1p(df[c])
    
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['SpendPerPerson'] = df['TotalSpend'] / (df['GroupSize'] + 1)
    
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,np.inf],
                         labels=['Child','Teen','Adult','Middle','Senior'],
                         include_lowest=True)
    
    df['DeckSide'] = df['CabinDeck'] + '_' + df['CabinSide']
    
    for col in ['CryoSleep','VIP']:
        df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0})\
                       .fillna(0).astype(int)
    
    # CAREFULLY SELECTED ADDITIONS (based on domain knowledge)
    # 1. CryoSleep-Spending interaction (should be very strong)
    df['CryoSpendFlag'] = (df['CryoSleep'] == 1) & (df['TotalSpend'] == 0)
    df['CryoSpendFlag'] = df['CryoSpendFlag'].astype(int)
    
    # 2. VIP-High spending pattern
    df['VIPHighSpender'] = (df['VIP'] == 1) & (df['TotalSpend'] > 1000)
    df['VIPHighSpender'] = df['VIPHighSpender'].astype(int)
    
    # 3. Group spending coherence
    group_spend_stats = df.groupby('GroupId')['TotalSpend'].agg(['mean', 'std']).reset_index()
    group_spend_stats.columns = ['GroupId', 'GroupSpendMean', 'GroupSpendStd']
    group_spend_stats['GroupSpendStd'] = group_spend_stats['GroupSpendStd'].fillna(0)
    df = df.merge(group_spend_stats, on='GroupId', how='left')
    
    # 4. Age-Group interaction
    df['FamilyWithKids'] = ((df['GroupSize'] > 1) & (df['Age'] < 18)).astype(int)
    
    return df

def precision_stacking():
    """
    Precision-tuned stacking ensemble
    """
    print("🎯 PRECISION TUNING - Final Push to Beat 80.8%")
    
    # Load data
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    # Enhanced feature engineering
    print("⚙️ Enhanced feature engineering...")
    train = enhanced_feature_engineering(train)
    test = enhanced_feature_engineering(test)
    
    # Your original clustering
    km_feats = train[['CabinNum','log_RoomService','log_FoodCourt',
                      'log_ShoppingMall','log_Spa','log_VRDeck','TotalSpend']].fillna(0)
    kmeans = KMeans(n_clusters=6, random_state=42).fit(km_feats)
    train['Cluster'] = kmeans.predict(km_feats)
    test['Cluster'] = kmeans.predict(test[km_feats.columns].fillna(0))
    
    # Your original advanced features
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
    
    test_drop_cols = [col for col in drop_cols if col in test.columns and col != 'Transported']
    X_test = test.drop(columns=test_drop_cols)
    
    # Categorical encoding
    cat_cols = ['AgeBin','CabinDeck','CabinSide','DeckSide','MaxSpendItem','Cluster']
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
        X_test[c] = X_test[c].astype('category').cat.codes
    
    print(f"📊 Features: {X.shape[1]}")
    
    # PRECISION-TUNED MODELS
    # Fine-tuned based on your success patterns
    
    # Model 1: Your best HGB with slight improvement
    hgb1 = HistGradientBoostingClassifier(
        learning_rate=0.045,  # Slightly lower for stability
        max_iter=600,         # More iterations
        max_leaf_nodes=31,
        min_samples_leaf=18,  # Slight regularization
        l2_regularization=0.05,
        random_state=42
    )
    
    # Model 2: Your second HGB with tuning
    hgb2 = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=400,
        max_leaf_nodes=28,    # Slightly smaller
        min_samples_leaf=15,
        l2_regularization=0.03,
        random_state=24
    )
    
    # Model 3: Your LGBM with improvements
    lgbm = LGBMClassifier(
        n_estimators=650,     # More estimators
        learning_rate=0.045,  # Slightly lower
        num_leaves=28,        # Slightly smaller
        min_child_samples=22, # More regularization
        subsample=0.85,       # Slight subsampling
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=0,
        verbose=-1
    )
    
    # Model 4: Precision-tuned XGBoost
    xgb = XGBClassifier(
        n_estimators=550,
        learning_rate=0.045,
        max_depth=5,          # Slightly shallower
        min_child_weight=4,   # More regularization
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Model 5: Add CatBoost for diversity
    cat = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False
    )
    
    # PRECISION STACKING
    stack = StackingClassifier(
        estimators=[
            ('hgb1', hgb1), ('hgb2', hgb2), 
            ('lgbm', lgbm), ('xgb', xgb), ('cat', cat)
        ],
        final_estimator=LogisticRegression(
            C=0.8,           # Slight regularization
            max_iter=2000,
            random_state=42
        ),
        cv=7,            # More robust CV
        passthrough=True,
        n_jobs=-1
    )
    
    # Cross-validation with more folds for precision
    print("📊 Precision cross-validation...")
    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    scores = cross_val_score(stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"🏆 Precision Ensemble Results:")
    print(f"   CV Accuracy: {scores.mean():.5f} ± {scores.std():.5f}")
    print(f"   Your baseline: 0.80804")
    print(f"   Improvement: {(scores.mean() - 0.80804)*100:+.2f}%")
    print(f"   Individual folds: {[f'{s:.5f}' for s in scores]}")
    
    # Train final model
    print("🚀 Training precision model...")
    stack.fit(X, y)
    preds_test = stack.predict(X_test).astype(bool)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': X_test.index,
        'Transported': preds_test
    })
    submission.to_csv('submission_precision.csv', index=False)
    
    print("✅ Created submission_precision.csv")
    
    # Model analysis
    print("\n🔍 Model Analysis:")
    
    # Test individual models
    models = {
        'HGB1': hgb1, 'HGB2': hgb2, 'LGBM': lgbm, 
        'XGB': xgb, 'CatBoost': cat
    }
    
    cv_simple = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        model_scores = cross_val_score(model, X, y, cv=cv_simple, scoring='accuracy', n_jobs=-1)
        print(f"{name}: {model_scores.mean():.5f} ± {model_scores.std():.5f}")
    
    return stack, scores.mean(), submission

def test_minimal_changes():
    """
    Test even more minimal changes - just hyperparameter tweaks
    """
    print("\n🔧 Testing Minimal Hyperparameter Changes...")
    
    # Your exact original approach
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    # Your exact original feature engineering
    def original_engineer(df):
        df = df.copy()
        df['GroupId'] = df.index.str.split('_').str[0]
        df['GroupSize'] = df.groupby('GroupId')['HomePlanet'].transform('size')
        df['IsAlone'] = (df['GroupSize']==1).astype(int)
        
        cabin = df['Cabin'].str.split('/', expand=True)
        df['CabinDeck'] = cabin[0].fillna('Unknown')
        cn = pd.to_numeric(cabin[1], errors='coerce')
        df['CabinNum'] = cn.fillna(cn.median())
        df['CabinSide'] = cabin[2].fillna('Unknown')
        
        spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
        df[spend_cols] = df[spend_cols].fillna(0)
        
        for c in spend_cols:
            df[f'log_{c}'] = np.log1p(df[c])
        
        df['TotalSpend'] = df[spend_cols].sum(axis=1)
        df['SpendPerPerson'] = df['TotalSpend'] / (df['GroupSize'] + 1)
        
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,np.inf],
                             labels=['Child','Teen','Adult','Middle','Senior'],
                             include_lowest=True)
        
        df['DeckSide'] = df['CabinDeck'] + '_' + df['CabinSide']
        
        for col in ['CryoSleep','VIP']:
            df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0})\
                           .fillna(0).astype(int)
        
        return df
    
    train = original_engineer(train)
    test = original_engineer(test)
    
    # Your exact clustering
    km_feats = train[['CabinNum','log_RoomService','log_FoodCourt',
                      'log_ShoppingMall','log_Spa','log_VRDeck','TotalSpend']].fillna(0)
    kmeans = KMeans(n_clusters=6, random_state=42).fit(km_feats)
    train['Cluster'] = kmeans.predict(km_feats)
    test['Cluster'] = kmeans.predict(test[km_feats.columns].fillna(0))
    
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
    
    test_drop_cols = [col for col in drop_cols if col in test.columns and col != 'Transported']
    X_test = test.drop(columns=test_drop_cols)
    
    cat_cols = ['AgeBin','CabinDeck','CabinSide','DeckSide','MaxSpendItem','Cluster']
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
        X_test[c] = X_test[c].astype('category').cat.codes
    
    # ONLY CHANGE: Slightly different random seeds and iterations
    hgb1 = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=550,  # +50 iterations
        max_leaf_nodes=31, random_state=42)
    
    hgb2 = HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=350,   # +50 iterations
        max_leaf_nodes=31, random_state=123)  # Different seed
    
    lgbm = LGBMClassifier(
        n_estimators=550, learning_rate=0.05,  # +50 estimators
        num_leaves=31, random_state=42, verbose=-1)  # Different seed
    
    stack = StackingClassifier(
        estimators=[('hgb1', hgb1), ('hgb2', hgb2), ('lgbm', lgbm)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, passthrough=True, n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"Minimal changes CV: {scores.mean():.5f}")
    print(f"vs your 0.80804: {(scores.mean() - 0.80804)*100:+.2f}%")
    
    # Create submission
    stack.fit(X, y)
    preds = stack.predict(X_test).astype(bool)
    submission = pd.DataFrame({
        'PassengerId': X_test.index,
        'Transported': preds
    })
    submission.to_csv('submission_minimal.csv', index=False)
    
    return scores.mean()

if __name__ == "__main__":
    print("🎯 FINAL PUSH - PRECISION TUNING")
    print("="*50)
    
    # Test minimal changes first
    minimal_score = test_minimal_changes()
    
    # Test precision approach
    model, precision_score, submission = precision_stacking()
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"Your original: 80.804%")
    print(f"Targeted: 80.757%")
    print(f"Minimal changes: {minimal_score*100:.3f}%")
    print(f"Precision tuned: {precision_score*100:.3f}%")
    
    best_approach = max([
        ("minimal", minimal_score),
        ("precision", precision_score)
    ], key=lambda x: x[1])
    
    print(f"\n🏆 Best approach: {best_approach[0]} with {best_approach[1]*100:.3f}%")
    
    if best_approach[1] > 0.80804:
        print(f"🎉 SUCCESS! Beat baseline by +{(best_approach[1]-0.80804)*100:.2f}%")
        print(f"📁 Best submission: submission_{best_approach[0]}.csv")
    else:
        print(f"⚠️ Close but not quite: {(best_approach[1]-0.80804)*100:+.2f}%")
    
    print(f"\n💡 All submissions created:")
    print(f"   📄 submission_minimal.csv")
    print(f"   📄 submission_precision.csv")
