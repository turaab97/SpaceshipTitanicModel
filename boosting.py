import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

"""
HYPERPARAMETER OPTIMIZATION
Going back to your exact 0.807 approach and just fine-tuning hyperparameters
No new features, no post-processing, just better hyperparameters
"""

def your_exact_feature_engineering(df):
    """
    Your EXACT feature engineering that got 0.807 - NO CHANGES
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

def hyperparameter_search():
    """
    Fine-tune hyperparameters for your exact setup
    """
    print("🎯 HYPERPARAMETER OPTIMIZATION")
    print("Using your exact 0.807 setup with better hyperparameters")
    print("="*50)
    
    # Load data
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    # Your exact feature engineering
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
    
    # Categorical encoding
    cat_cols = ['AgeBin','CabinDeck','CabinSide','DeckSide','MaxSpendItem','Cluster']
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
        X_test[c] = X_test[c].astype('category').cat.codes
    
    print(f"📊 Features: {X.shape[1]} (your exact feature set)")
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test 1: Your original hyperparameters
    print("\n📊 Testing your original hyperparameters...")
    
    hgb1_orig = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500,
        max_leaf_nodes=31, random_state=42)
    
    hgb2_orig = HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=300,
        max_leaf_nodes=31, random_state=24)
    
    lgbm_orig = LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, random_state=0, verbose=-1)
    
    xgb_orig = XGBClassifier(
        n_estimators=500, learning_rate=0.05,
        max_depth=6, random_state=42, eval_metric='logloss')
    
    stack_orig = StackingClassifier(
        estimators=[
            ('hgb1', hgb1_orig), ('hgb2', hgb2_orig),
            ('lgbm', lgbm_orig), ('xgb', xgb_orig)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, passthrough=True, n_jobs=-1
    )
    
    scores_orig = cross_val_score(stack_orig, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Original parameters: {scores_orig.mean():.4f} ± {scores_orig.std():.4f}")
    
    # Test 2: Slightly tuned hyperparameters
    print("\n📊 Testing tuned hyperparameters...")
    
    # Small adjustments that often help
    hgb1_tuned = HistGradientBoostingClassifier(
        learning_rate=0.04,  # Slightly lower
        max_iter=600,  # More iterations
        max_leaf_nodes=35,  # Slightly more complex
        min_samples_leaf=15,  # Some regularization
        random_state=42)
    
    hgb2_tuned = HistGradientBoostingClassifier(
        learning_rate=0.08,  # Slightly lower
        max_iter=400,  # More iterations
        max_leaf_nodes=35,  # Slightly more complex
        min_samples_leaf=15,  # Some regularization
        random_state=24)
    
    lgbm_tuned = LGBMClassifier(
        n_estimators=600,  # More trees
        learning_rate=0.04,  # Slightly lower
        num_leaves=35,  # Slightly more complex
        min_child_samples=15,  # Some regularization
        subsample=0.95,  # Very slight subsampling
        colsample_bytree=0.95,  # Very slight feature sampling
        random_state=0, verbose=-1)
    
    xgb_tuned = XGBClassifier(
        n_estimators=600,  # More trees
        learning_rate=0.04,  # Slightly lower
        max_depth=7,  # Slightly deeper
        min_child_weight=2,  # Some regularization
        subsample=0.95,  # Very slight subsampling
        colsample_bytree=0.95,  # Very slight feature sampling
        random_state=42, eval_metric='logloss')
    
    stack_tuned = StackingClassifier(
        estimators=[
            ('hgb1', hgb1_tuned), ('hgb2', hgb2_tuned),
            ('lgbm', lgbm_tuned), ('xgb', xgb_tuned)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, passthrough=True, n_jobs=-1
    )
    
    scores_tuned = cross_val_score(stack_tuned, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Tuned parameters: {scores_tuned.mean():.4f} ± {scores_tuned.std():.4f}")
    
    # Test 3: Different meta-learner
    print("\n📊 Testing different meta-learners...")
    
    # Try Ridge instead of Logistic Regression
    from sklearn.linear_model import RidgeClassifier
    
    stack_ridge = StackingClassifier(
        estimators=[
            ('hgb1', hgb1_orig), ('hgb2', hgb2_orig),
            ('lgbm', lgbm_orig), ('xgb', xgb_orig)
        ],
        final_estimator=RidgeClassifier(alpha=1.0),
        cv=5, passthrough=True, n_jobs=-1
    )
    
    scores_ridge = cross_val_score(stack_ridge, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Ridge meta-learner: {scores_ridge.mean():.4f} ± {scores_ridge.std():.4f}")
    
    # Test 4: Different CV strategy for stacking
    print("\n📊 Testing different CV strategies...")
    
    stack_cv3 = StackingClassifier(
        estimators=[
            ('hgb1', hgb1_orig), ('hgb2', hgb2_orig),
            ('lgbm', lgbm_orig), ('xgb', xgb_orig)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,  # Less folds might reduce overfitting
        passthrough=True, n_jobs=-1
    )
    
    scores_cv3 = cross_val_score(stack_cv3, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"3-fold stacking: {scores_cv3.mean():.4f} ± {scores_cv3.std():.4f}")
    
    # Find best configuration
    all_configs = {
        'Original': (stack_orig, scores_orig.mean()),
        'Tuned': (stack_tuned, scores_tuned.mean()),
        'Ridge': (stack_ridge, scores_ridge.mean()),
        '3-fold': (stack_cv3, scores_cv3.mean())
    }
    
    best_name = max(all_configs, key=lambda x: all_configs[x][1])
    best_model, best_score = all_configs[best_name]
    
    print(f"\n🏆 Best configuration: {best_name} with CV: {best_score:.4f}")
    
    # Train best model
    print(f"\n🚀 Training {best_name} configuration...")
    best_model.fit(X, y)
    
    # Make predictions
    preds = best_model.predict(X_test).astype(bool)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': X_test.index,
        'Transported': preds
    })
    submission['PassengerId'] = submission['PassengerId'].astype(str)
    submission.to_csv('submission_hyperopt.csv', index=False)
    
    print("\n✅ Created submission_hyperopt.csv")
    
    # Additional: Test removing each model
    print("\n🔍 Testing model importance...")
    
    model_names = ['hgb1', 'hgb2', 'lgbm', 'xgb']
    base_models = [hgb1_orig, hgb2_orig, lgbm_orig, xgb_orig]
    
    for i, remove_name in enumerate(model_names):
        remaining_models = [(name, model) for j, (name, model) in enumerate(zip(model_names, base_models)) if j != i]
        
        stack_removed = StackingClassifier(
            estimators=remaining_models,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5, passthrough=True, n_jobs=-1
        )
        
        scores_removed = cross_val_score(stack_removed, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"Without {remove_name}: {scores_removed.mean():.4f} (diff: {(scores_removed.mean() - scores_orig.mean())*100:+.3f}%)")
    
    return best_model, best_score

if __name__ == "__main__":
    print("🎯 SPACESHIP TITANIC - HYPERPARAMETER OPTIMIZATION")
    print("Back to basics: Your exact 0.807 approach with fine-tuning")
    print("="*50)
    
    model, score = hyperparameter_search()
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"Your baseline: 0.8070")
    print(f"All feature engineering attempts: Failed (0.8036, 0.8013, 0.8066)")
    print(f"Post-processing attempt: Failed badly (0.7992)")
    print(f"This hyperopt approach: {score:.4f}")
    
    print("\n💡 Key insights:")
    print("1. Your original feature engineering is already excellent")
    print("2. Adding complexity (features/rules) makes it worse")
    print("3. Small hyperparameter adjustments might help")
    print("4. Sometimes the simplest approach is best")
    
    if score > 0.807:
        print(f"\n🎉 Finally! Improved by {(score - 0.807)*100:+.2f}%")
    else:
        print("\n🤔 If this doesn't work, try:")
        print("1. Different random seeds (set random_state to 0, 123, etc)")
        print("2. Pseudo-labeling with high-confidence test predictions")
        print("3. Averaging multiple runs with different seeds")
