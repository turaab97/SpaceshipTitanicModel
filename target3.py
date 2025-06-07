import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

"""
MINIMAL BOOST STRATEGY
Since the enhanced version (0.80360) performed worse than baseline (0.807),
we'll be much more conservative. Only add features that:
1. Have clear logical relationships
2. Don't increase complexity too much
3. Focus on fixing known issues in the data
"""

def minimal_boost_engineering(df):
    """
    Your EXACT original features + only 2-3 critical additions
    """
    df = df.copy()
    
    # ========== YOUR EXACT ORIGINAL FEATURES ==========
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
    
    # Your original interaction features
    df['CryoSpendInteraction'] = df['CryoSleep'] * df['TotalSpend']
    df['VIPSpendInteraction'] = df['VIP'] * df['TotalSpend']
    df['YoungAloneFlag'] = ((df['Age'] < 25) & (df['IsAlone'] == 1)).astype(int)
    
    # ========== MINIMAL HIGH-IMPACT ADDITIONS ==========
    
    # 1. CryoSleep validation (this is a hard rule in the problem)
    # People in CryoSleep should have 0 spending
    df['CryoSleepValid'] = ((df['CryoSleep'] == 1) & (df['TotalSpend'] == 0)).astype(int)
    
    # 2. Simple spending pattern
    df['HasAnySpending'] = (df['TotalSpend'] > 0).astype(int)
    
    # 3. Luxury vs Essential spending
    df['LuxurySpendRatio'] = (df['Spa'] + df['VRDeck']) / (df['TotalSpend'] + 1)
    
    return df

def create_minimal_boost_model():
    """
    Conservative improvement approach
    """
    print("🎯 MINIMAL BOOST STRATEGY")
    print("Conservative improvements to your 0.807 baseline")
    print("="*50)
    
    # Load data
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    # Apply minimal feature engineering
    print("⚙️ Applying minimal feature engineering...")
    train = minimal_boost_engineering(train)
    test = minimal_boost_engineering(test)
    
    # Your EXACT original clustering
    print("🔮 Creating clusters (your original approach)...")
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
    
    print(f"📊 Total features: {X.shape[1]} (only +3 from your original)")
    
    # ========== APPROACH 1: Your exact models with minor tuning ==========
    print("\n🔧 Testing Approach 1: Minor hyperparameter tuning...")
    
    hgb1 = HistGradientBoostingClassifier(
        learning_rate=0.05, 
        max_iter=550,  # Slightly more iterations
        max_leaf_nodes=31, 
        min_samples_leaf=20,  # Add slight regularization
        random_state=42
    )
    
    hgb2 = HistGradientBoostingClassifier(
        learning_rate=0.1, 
        max_iter=350,  # Slightly more iterations
        max_leaf_nodes=31,
        min_samples_leaf=20,  # Add slight regularization
        random_state=24
    )
    
    lgbm = LGBMClassifier(
        n_estimators=550,  # Slightly more trees
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,  # Add regularization
        subsample=0.9,  # Slight subsampling
        random_state=0, 
        verbose=-1
    )
    
    xgb = XGBClassifier(
        n_estimators=550,  # Slightly more trees
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,  # Add regularization
        subsample=0.9,  # Slight subsampling
        random_state=42, 
        eval_metric='logloss'
    )
    
    # Your original stacking
    stack1 = StackingClassifier(
        estimators=[
            ('hgb1', hgb1), ('hgb2', hgb2),
            ('lgbm', lgbm), ('xgb', xgb)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores1 = cross_val_score(stack1, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"Approach 1 CV: {scores1.mean():.4f} ({scores1.mean()*100:.1f}%)")
    
    # ========== APPROACH 2: Remove XGBoost (sometimes less is more) ==========
    print("\n🔧 Testing Approach 2: Without XGBoost...")
    
    stack2 = StackingClassifier(
        estimators=[
            ('hgb1', hgb1), ('hgb2', hgb2), ('lgbm', lgbm)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    scores2 = cross_val_score(stack2, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Approach 2 CV: {scores2.mean():.4f} ({scores2.mean()*100:.1f}%)")
    
    # ========== APPROACH 3: Weighted voting ==========
    print("\n🔧 Testing Approach 3: Weighted voting ensemble...")
    
    from sklearn.ensemble import VotingClassifier
    
    voting = VotingClassifier(
        estimators=[
            ('hgb1', hgb1), ('hgb2', hgb2),
            ('lgbm', lgbm), ('xgb', xgb)
        ],
        voting='soft',
        weights=[1.0, 1.0, 1.2, 1.1],  # Slightly favor LGBM and XGB
        n_jobs=-1
    )
    
    scores3 = cross_val_score(voting, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Approach 3 CV: {scores3.mean():.4f} ({scores3.mean()*100:.1f}%)")
    
    # ========== APPROACH 4: Single best model ==========
    print("\n🔧 Testing Approach 4: Single best model...")
    
    # Test individual models
    individual_scores = {}
    for name, model in [('HGB1', hgb1), ('HGB2', hgb2), ('LGBM', lgbm), ('XGB', xgb)]:
        ind_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        individual_scores[name] = ind_scores.mean()
        print(f"{name}: {ind_scores.mean():.4f}")
    
    # Find best approach
    all_scores = {
        'Stacking with XGB': scores1.mean(),
        'Stacking without XGB': scores2.mean(),
        'Weighted Voting': scores3.mean(),
        **{f'Single {k}': v for k, v in individual_scores.items()}
    }
    
    best_approach = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_approach]
    
    print(f"\n🏆 Best approach: {best_approach} with CV score: {best_score:.4f}")
    print(f"   vs your baseline: {(best_score - 0.807)*100:+.2f}%")
    
    # Train and create submission with best approach
    print(f"\n🚀 Training {best_approach}...")
    
    if best_approach == 'Stacking with XGB':
        final_model = stack1
    elif best_approach == 'Stacking without XGB':
        final_model = stack2
    elif best_approach == 'Weighted Voting':
        final_model = voting
    else:
        # Single model
        model_name = best_approach.split()[-1]
        final_model = {'HGB1': hgb1, 'HGB2': hgb2, 'LGBM': lgbm, 'XGB': xgb}[model_name]
    
    final_model.fit(X, y)
    preds = final_model.predict(X_test).astype(bool)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': X_test.index,
        'Transported': preds
    })
    submission['PassengerId'] = submission['PassengerId'].astype(str)
    submission.to_csv('submission_minimal_boost.csv', index=False)
    
    print("✅ Created submission_minimal_boost.csv")
    
    # Additional analysis
    print("\n📊 Feature impact analysis:")
    
    # Check CryoSleep rule
    cryo_valid = train[(train['CryoSleep']==1) & (train['CryoSleepValid']==1)]
    cryo_invalid = train[(train['CryoSleep']==1) & (train['CryoSleepValid']==0)]
    print(f"CryoSleep passengers following spending rule: {len(cryo_valid)} ({len(cryo_valid)/(len(cryo_valid)+len(cryo_invalid))*100:.1f}%)")
    print(f"CryoSleep rule violations: {len(cryo_invalid)}")
    
    # Test without the new features to see impact
    print("\n🔍 Testing impact of new features...")
    X_no_new = X.drop(columns=['CryoSleepValid', 'HasAnySpending', 'LuxurySpendRatio'])
    X_test_no_new = X_test.drop(columns=['CryoSleepValid', 'HasAnySpending', 'LuxurySpendRatio'])
    
    scores_no_new = cross_val_score(stack1, X_no_new, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Without new features: {scores_no_new.mean():.4f}")
    print(f"With new features: {scores1.mean():.4f}")
    print(f"Feature impact: {(scores1.mean() - scores_no_new.mean())*100:+.2f}%")
    
    return final_model, best_score

def test_pure_baseline():
    """
    Test your EXACT original code to verify 0.807 baseline
    """
    print("\n🔬 Verifying your original 0.807 baseline...")
    
    # Your exact original code
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    # Your exact feature engineering
    train['GroupId'] = train.index.str.split('_').str[0]
    train['GroupSize'] = train.groupby('GroupId')['HomePlanet'].transform('size')
    train['IsAlone'] = (train['GroupSize']==1).astype(int)
    
    cabin = train['Cabin'].str.split('/', expand=True)
    train['CabinDeck'] = cabin[0].fillna('Unknown')
    cn = pd.to_numeric(cabin[1], errors='coerce')
    train['CabinNum'] = cn.fillna(cn.median())
    train['CabinSide'] = cabin[2].fillna('Unknown')
    
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    train[spend_cols] = train[spend_cols].fillna(0)
    
    for c in spend_cols:
        train[f'log_{c}'] = np.log1p(train[c])
    
    train['TotalSpend'] = train[spend_cols].sum(axis=1)
    train['SpendPerPerson'] = train['TotalSpend'] / (train['GroupSize'] + 1)
    
    train['Age'] = train['Age'].fillna(train['Age'].median())
    train['AgeBin'] = pd.cut(train['Age'], bins=[0,12,18,35,60,np.inf],
                           labels=['Child','Teen','Adult','Middle','Senior'],
                           include_lowest=True)
    
    train['DeckSide'] = train['CabinDeck'] + '_' + train['CabinSide']
    
    for col in ['CryoSleep','VIP']:
        train[col] = train[col].map({True:1, False:0, 'True':1, 'False':0})\
                          .fillna(0).astype(int)
    
    train['CryoSpendInteraction'] = train['CryoSleep'] * train['TotalSpend']
    train['VIPSpendInteraction'] = train['VIP'] * train['TotalSpend']
    train['YoungAloneFlag'] = ((train['Age'] < 25) & (train['IsAlone'] == 1)).astype(int)
    
    # Your exact clustering
    km_feats = train[['CabinNum','log_RoomService','log_FoodCourt',
                      'log_ShoppingMall','log_Spa','log_VRDeck','TotalSpend']].fillna(0)
    kmeans = KMeans(n_clusters=6, random_state=42).fit(km_feats)
    train['Cluster'] = kmeans.predict(km_feats)
    
    train['MaxSpendItem'] = train[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].idxmax(axis=1)
    
    for col in ['HomePlanet','Destination','DeckSide']:
        freq = train[col].value_counts() / len(train)
        train[f'{col}_Freq'] = train[col].map(freq).fillna(0)
    
    # Your exact data prep
    y = train['Transported'].astype(int)
    drop_cols = ['Transported','Name','Cabin','GroupId','HomePlanet','Destination']
    X = train.drop(columns=drop_cols)
    
    cat_cols = ['AgeBin','CabinDeck','CabinSide','DeckSide','MaxSpendItem','Cluster']
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
    
    # Your exact models
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
    
    stack = StackingClassifier(
        estimators=[
            ('hgb1', hgb1), ('hgb2', hgb2),
            ('lgbm', lgbm), ('xgb', xgb)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"Pure baseline CV: {scores.mean():.4f} (should be ~0.807)")
    
    return scores.mean()

if __name__ == "__main__":
    print("🎯 MINIMAL BOOST STRATEGY")
    print("Learning from the enhanced approach failure")
    print("="*50)
    
    # First verify baseline
    baseline_score = test_pure_baseline()
    
    # Then run minimal boost
    model, best_score = create_minimal_boost_model()
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"Your baseline: {baseline_score:.4f}")
    print(f"Enhanced attempt: 0.8036 (worse)")
    print(f"Minimal boost: {best_score:.4f}")
    print(f"Net improvement: {(best_score - baseline_score)*100:+.2f}%")
    
    print("\n💡 Key insights:")
    print("1. The enhanced approach had too many features (overfitting)")
    print("2. Sometimes less is more - minimal changes work better")
    print("3. Focus on high-impact features like CryoSleep validation")
    print("4. Different ensemble strategies can make a difference")
