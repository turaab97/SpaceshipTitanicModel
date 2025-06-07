import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier  # Adding just ONE new model
from sklearn.linear_model import LogisticRegression

"""
TARGETED IMPROVEMENT APPROACH
Goal: Take your successful 80.8% approach and make MINIMAL changes for 81-82%

Strategy:
1. Keep your exact feature engineering that worked
2. Add just ONE new model (XGBoost) to your ensemble  
3. Add only 3-5 critical new features
4. Slightly tune hyperparameters
5. NO major changes to data cleaning

This should be safer than the over-engineered approach
"""

def minimal_feature_engineering(df):
    """
    Your original feature engineering + just a few critical additions
    """
    df = df.copy()
    
    # YOUR ORIGINAL FEATURES (keep exactly as they were)
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
    
    # Your original log features
    for c in spend_cols:
        df[f'log_{c}'] = np.log1p(df[c])
    
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['SpendPerPerson'] = df['TotalSpend'] / (df['GroupSize'] + 1)
    
    # Your original age features
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,np.inf],
                         labels=['Child','Teen','Adult','Middle','Senior'],
                         include_lowest=True)
    
    # Your original composite features
    df['DeckSide'] = df['CabinDeck'] + '_' + df['CabinSide']
    
    # Your original boolean conversion
    for col in ['CryoSleep','VIP']:
        df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0})\
                       .fillna(0).astype(int)
    
    # MINIMAL ADDITIONS (only 3 new features)
    # 1. CryoSleep spending interaction (this should be very predictive)
    df['CryoSpendInteraction'] = df['CryoSleep'] * df['TotalSpend']
    
    # 2. VIP spending interaction
    df['VIPSpendInteraction'] = df['VIP'] * df['TotalSpend']
    
    # 3. Age-Alone interaction (young people alone behave differently)
    df['YoungAloneFlag'] = ((df['Age'] < 25) & (df['IsAlone'] == 1)).astype(int)
    
    return df

def targeted_stacking_improvement():
    """
    Your original stacking approach + minimal improvements
    """
    print("🚀 Targeted Improvement: Building on Your 80.8% Success")
    print("Strategy: Minimal changes to your working approach")
    
    # Load data exactly like your original
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    # Apply minimal feature engineering
    print("⚙️ Applying your original + 3 new features...")
    train = minimal_feature_engineering(train)
    test = minimal_feature_engineering(test)
    
    # Your original clustering (keep exactly the same)
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
    
    # Prepare data exactly like your original
    y = train['Transported'].astype(int)
    drop_cols = ['Transported','Name','Cabin','GroupId','HomePlanet','Destination']
    X = train.drop(columns=drop_cols)
    
    # For test set, only drop columns that exist (test doesn't have 'Transported')
    test_drop_cols = [col for col in drop_cols if col in test.columns and col != 'Transported']
    X_test = test.drop(columns=test_drop_cols)
    
    # Your original categorical encoding
    cat_cols = ['AgeBin','CabinDeck','CabinSide','DeckSide','MaxSpendItem','Cluster']
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
        X_test[c] = X_test[c].astype('category').cat.codes
    
    print(f"📊 Features: {X.shape[1]} (vs your original ~25)")
    
    # YOUR ORIGINAL MODELS (keep exactly the same)
    hgb1 = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500,
        max_leaf_nodes=31, random_state=42)
    
    hgb2 = HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=300,
        max_leaf_nodes=31, random_state=24)
    
    lgbm = LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, random_state=0, verbose=-1)
    
    # MINIMAL ADDITION: Just add XGBoost (similar hyperparameters)
    xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.05,
        max_depth=6, random_state=42, eval_metric='logloss')
    
    # Enhanced stacking: Your 3 models + 1 new model
    stack = StackingClassifier(
        estimators=[
            ('hgb1', hgb1), ('hgb2', hgb2),
            ('lgbm', lgbm), ('xgb', xgb)  # Just adding XGBoost
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    # Cross-validation
    print("📊 Cross-validating improved ensemble...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"🎯 Improved Ensemble CV Results:")
    print(f"   Your original: 0.8081 (80.8%)")
    print(f"   This approach: {scores.mean():.4f} ({scores.mean()*100:.1f}%)")
    print(f"   Improvement: +{(scores.mean() - 0.8081)*100:.2f}%")
    print(f"   Std deviation: ±{scores.std():.4f}")
    
    # Train and predict
    print("🚀 Training final model...")
    stack.fit(X, y)
    preds_test = stack.predict(X_test).astype(bool)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': X_test.index,
        'Transported': preds_test
    })
    submission.to_csv('submission_targeted.csv', index=False)
    
    print("✅ Created submission_targeted.csv")
    
    # Feature importance analysis
    print("\n🔍 Feature Impact Analysis:")
    
    # Compare your original 3-model vs new 4-model
    original_stack = StackingClassifier(
        estimators=[('hgb1', hgb1), ('hgb2', hgb2), ('lgbm', lgbm)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, passthrough=True, n_jobs=-1
    )
    
    original_scores = cross_val_score(original_stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"Your original 3-model ensemble: {original_scores.mean():.4f}")
    print(f"New 4-model ensemble: {scores.mean():.4f}")
    print(f"Ensemble improvement: +{(scores.mean() - original_scores.mean())*100:.2f}%")
    
    # Test individual model contributions
    print("\n🔍 Individual Model Performance:")
    models = {'HGB1': hgb1, 'HGB2': hgb2, 'LGBM': lgbm, 'XGB': xgb}
    
    for name, model in models.items():
        model_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"{name}: {model_scores.mean():.4f} ± {model_scores.std():.4f}")
    
    return stack, scores.mean(), submission

def create_conservative_ensemble():
    """
    Even more conservative: Your exact original + just better hyperparameters
    """
    print("\n🔧 Testing Conservative Approach: Just Better Hyperparameters")
    
    # Load and process exactly like your original
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
    
    # Your exact original clustering and features
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
    
    # For test set, only drop columns that actually exist
    test_drop_cols = [col for col in drop_cols if col in test.columns and col != 'Transported']
    X_test = test.drop(columns=test_drop_cols)
    
    cat_cols = ['AgeBin','CabinDeck','CabinSide','DeckSide','MaxSpendItem','Cluster']
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
        X_test[c] = X_test[c].astype('category').cat.codes
    
    # ONLY CHANGE: Slightly better hyperparameters
    hgb1 = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=600,  # +100 iterations
        max_leaf_nodes=31, random_state=42)
    
    hgb2 = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=400,  # Slight change
        max_leaf_nodes=31, random_state=24)
    
    lgbm = LGBMClassifier(
        n_estimators=600, learning_rate=0.05,  # +100 estimators
        num_leaves=31, random_state=0, verbose=-1)
    
    # Your exact original stacking
    stack = StackingClassifier(
        estimators=[('hgb1', hgb1), ('hgb2', hgb2), ('lgbm', lgbm)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    # Test
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"Conservative approach CV: {scores.mean():.4f}")
    print(f"vs your original 0.8081: +{(scores.mean() - 0.8081)*100:.2f}%")
    
    return scores.mean()

if __name__ == "__main__":
    print("🎯 TARGETED IMPROVEMENT STRATEGY")
    print("Goal: Beat your 80.8% with minimal risk")
    print("="*50)
    
    # Test conservative approach first
    conservative_score = create_conservative_ensemble()
    
    # Test targeted approach
    model, targeted_score, submission = targeted_stacking_improvement()
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"Your original baseline: 80.8%")
    print(f"Advanced approach: 78.8% (failed)")
    print(f"Conservative approach: {conservative_score*100:.1f}%")
    print(f"Targeted approach: {targeted_score*100:.1f}%")
    
    if targeted_score > 0.8081:
        print(f"\n🎉 SUCCESS! Targeted approach beats baseline by +{(targeted_score-0.8081)*100:.2f}%")
    else:
        print(f"\n⚠️ Targeted approach: +{(targeted_score-0.8081)*100:.2f}% vs baseline")
    
    print(f"\n📁 Created: submission_targeted.csv")
    print(f"🎯 Strategy: Minimal changes to your working approach")
