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
SEED AVERAGING STRATEGY
Using your successful tuned hyperparameters with multiple random seeds
This reduces variance and often improves leaderboard score
"""

def your_exact_feature_engineering(df):
    """
    Your EXACT feature engineering - NO CHANGES
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

def seed_averaging_ensemble():
    """
    Train with multiple seeds and average predictions
    """
    print("🎯 SEED AVERAGING STRATEGY")
    print("Using your successful tuned hyperparameters with multiple seeds")
    print("="*50)
    
    # Load data
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    # Your exact feature engineering
    print("⚙️ Applying your exact feature engineering...")
    train = your_exact_feature_engineering(train)
    test = your_exact_feature_engineering(test)
    
    # Prepare data (same as before)
    y = train['Transported'].astype(int)
    
    # Multiple seeds for more stable predictions
    seeds = [42, 0, 123, 999, 2024, 7, 2023, 555, 888, 314]
    
    all_predictions = []
    all_probabilities = []
    cv_scores = []
    
    print(f"\n🎲 Training with {len(seeds)} different random seeds...")
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*50}")
        print(f"Seed {i+1}/{len(seeds)}: {seed}")
        
        # Your exact clustering with current seed
        km_feats = train[['CabinNum','log_RoomService','log_FoodCourt',
                          'log_ShoppingMall','log_Spa','log_VRDeck','TotalSpend']].fillna(0)
        kmeans = KMeans(n_clusters=6, random_state=seed).fit(km_feats)
        train_seed = train.copy()
        test_seed = test.copy()
        train_seed['Cluster'] = kmeans.predict(km_feats)
        test_seed['Cluster'] = kmeans.predict(test_seed[km_feats.columns].fillna(0))
        
        # Your exact advanced features
        train_seed['MaxSpendItem'] = train_seed[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].idxmax(axis=1)
        test_seed['MaxSpendItem'] = test_seed[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].idxmax(axis=1)
        
        for col in ['HomePlanet','Destination','DeckSide']:
            freq = train_seed[col].value_counts() / len(train_seed)
            train_seed[f'{col}_Freq'] = train_seed[col].map(freq).fillna(0)
            test_seed[f'{col}_Freq'] = test_seed[col].map(freq).fillna(0)
        
        # Prepare data
        drop_cols = ['Transported','Name','Cabin','GroupId','HomePlanet','Destination']
        X = train_seed.drop(columns=drop_cols)
        X_test = test_seed.drop(columns=[c for c in drop_cols if c in test_seed.columns])
        
        # Categorical encoding
        cat_cols = ['AgeBin','CabinDeck','CabinSide','DeckSide','MaxSpendItem','Cluster']
        for c in cat_cols:
            X[c] = X[c].astype('category').cat.codes
            X_test[c] = X_test[c].astype('category').cat.codes
        
        # Create models with TUNED hyperparameters (that got you 0.80827)
        hgb1 = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=600,
            max_leaf_nodes=35,
            min_samples_leaf=15,
            random_state=seed
        )
        
        hgb2 = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=400,
            max_leaf_nodes=35,
            min_samples_leaf=15,
            random_state=seed
        )
        
        lgbm = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.04,
            num_leaves=35,
            min_child_samples=15,
            subsample=0.95,
            colsample_bytree=0.95,
            random_state=seed,
            verbose=-1
        )
        
        xgb = XGBClassifier(
            n_estimators=600,
            learning_rate=0.04,
            max_depth=7,
            min_child_weight=2,
            subsample=0.95,
            colsample_bytree=0.95,
            random_state=seed,
            eval_metric='logloss'
        )
        
        # Create stacking ensemble
        stack = StackingClassifier(
            estimators=[
                ('hgb1', hgb1), ('hgb2', hgb2),
                ('lgbm', lgbm), ('xgb', xgb)
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
            cv=5,
            passthrough=True,
            n_jobs=-1
        )
        
        # Cross-validation for this seed
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_scores.append(scores.mean())
        print(f"CV Score: {scores.mean():.4f} ± {scores.std():.4f}")
        
        # Train model
        print(f"Training model with seed {seed}...")
        stack.fit(X, y)
        
        # Get predictions
        pred_proba = stack.predict_proba(X_test)[:, 1]
        pred = stack.predict(X_test)
        
        all_probabilities.append(pred_proba)
        all_predictions.append(pred)
        
        print(f"Predictions range: [{pred_proba.min():.3f}, {pred_proba.max():.3f}]")
        print(f"Positive rate: {pred.mean()*100:.1f}%")
    
    # Average predictions
    print("\n" + "="*50)
    print("📊 AGGREGATING PREDICTIONS")
    print("="*50)
    
    # Convert to numpy arrays
    all_probabilities = np.array(all_probabilities)
    all_predictions = np.array(all_predictions)
    np.save('all_predictions.npy', all_predictions)
    # Method 1: Average probabilities
    avg_probabilities = all_probabilities.mean(axis=0)
    avg_predictions = (avg_probabilities >= 0.5).astype(bool)
    
    # Method 2: Voting (majority)
    vote_predictions = (all_predictions.mean(axis=0) >= 0.5).astype(bool)
    
    # Method 3: Weighted average (weight by CV score)
    cv_scores = np.array(cv_scores)
    weights = cv_scores / cv_scores.sum()
    weighted_probabilities = np.average(all_probabilities, axis=0, weights=weights)
    weighted_predictions = (weighted_probabilities >= 0.5).astype(bool)
    
    # Statistics
    print(f"\nCV Scores across seeds:")
    print(f"  Mean: {cv_scores.mean():.4f}")
    print(f"  Std:  {cv_scores.std():.4f}")
    print(f"  Min:  {cv_scores.min():.4f}")
    print(f"  Max:  {cv_scores.max():.4f}")
    
    print(f"\nPrediction agreement:")
    print(f"  All seeds agree: {(all_predictions.std(axis=0) == 0).sum()} / {len(avg_predictions)} ({(all_predictions.std(axis=0) == 0).mean()*100:.1f}%)")
    print(f"  High confidence (prob < 0.3 or > 0.7): {((avg_probabilities < 0.3) | (avg_probabilities > 0.7)).sum()} ({((avg_probabilities < 0.3) | (avg_probabilities > 0.7)).mean()*100:.1f}%)")
    
    print(f"\nDifferences between methods:")
    print(f"  Avg vs Vote: {(avg_predictions != vote_predictions).sum()} different")
    print(f"  Avg vs Weighted: {(avg_predictions != weighted_predictions).sum()} different")
    
    # Create submissions
    print("\n📁 Creating submissions...")
    
    # Submission 1: Average probabilities
    submission_avg = pd.DataFrame({
        'PassengerId': X_test.index,
        'Transported': avg_predictions
    })
    submission_avg['PassengerId'] = submission_avg['PassengerId'].astype(str)
    submission_avg.to_csv('submission_seed_avg.csv', index=False)
    print("✅ Created submission_seed_avg.csv (probability averaging)")
    
    # Submission 2: Majority voting
    submission_vote = pd.DataFrame({
        'PassengerId': X_test.index,
        'Transported': vote_predictions
    })
    submission_vote['PassengerId'] = submission_vote['PassengerId'].astype(str)
    submission_vote.to_csv('submission_seed_vote.csv', index=False)
    print("✅ Created submission_seed_vote.csv (majority voting)")
    
    # Submission 3: Weighted by CV score
    submission_weighted = pd.DataFrame({
        'PassengerId': X_test.index,
        'Transported': weighted_predictions
    })
    submission_weighted['PassengerId'] = submission_weighted['PassengerId'].astype(str)
    submission_weighted.to_csv('submission_seed_weighted.csv', index=False)
    print("✅ Created submission_seed_weighted.csv (CV-weighted averaging)")
    
    # Test different thresholds on averaged probabilities
    print("\n🎯 Testing different thresholds on averaged probabilities:")
    thresholds = [0.48, 0.49, 0.50, 0.51, 0.52]
    
    best_threshold = 0.5
    if cv_scores.mean() > 0.808:  # Only if we're doing better
        print("\nSince CV is good, testing alternative thresholds...")
        for thresh in thresholds:
            thresh_pred = (avg_probabilities >= thresh).astype(bool)
            transported_rate = thresh_pred.mean()
            print(f"  Threshold {thresh}: {transported_rate*100:.1f}% transported")
            
            if thresh != 0.5:
                submission_thresh = pd.DataFrame({
                    'PassengerId': X_test.index,
                    'Transported': thresh_pred
                })
                submission_thresh['PassengerId'] = submission_thresh['PassengerId'].astype(str)
                submission_thresh.to_csv(f'submission_thresh_{thresh}.csv', index=False)
    
    return avg_probabilities, cv_scores

if __name__ == "__main__":
    print("🎯 SPACESHIP TITANIC - SEED AVERAGING")
    print("Building on your 0.80827 success")
    print("="*50)
    
    avg_proba, cv_scores = seed_averaging_ensemble()
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"Your baseline: 0.8070")
    print(f"Hyperparameter tuning: 0.8083")
    print(f"This seed averaging CV: {cv_scores.mean():.4f}")
    print(f"Expected LB improvement: +{(cv_scores.mean() - 0.8083)*100:.2f}%")
    
    print("\n💡 Recommendations:")
    print("1. Try submission_seed_avg.csv first (usually best)")
    print("2. If that doesn't improve, try submission_seed_weighted.csv")
    print("3. submission_seed_vote.csv as backup")
    
    print("\n🎲 Why seed averaging works:")
    print("- Reduces overfitting to validation set")
    print("- Captures different local optima")
    print("- More stable predictions")
    print("- Often gains 0.001-0.003 on leaderboard")
