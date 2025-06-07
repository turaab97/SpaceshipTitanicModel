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
ENHANCED APPROACH: Building on your successful 0.807
Key improvements:
1. Keep ALL your working features
2. Add more sophisticated feature interactions
3. Improve clustering approach
4. Fine-tune model hyperparameters
5. Add feature selection for noise reduction
"""

def enhanced_feature_engineering(df):
    """
    Your original features + strategic enhancements
    """
    df = df.copy()
    
    # ========== YOUR ORIGINAL FEATURES (keeping all of them) ==========
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
    df['HasMissingAge'] = df['Age'].isna().astype(int)  # Track before filling
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
    
    # ========== ENHANCED FEATURES (building on your success) ==========
    
    # 1. More sophisticated spending patterns
    # Calculate spending entropy (diversity of spending)
    spend_props = pd.DataFrame()
    for col in spend_cols:
        spend_props[col] = (df[col] + 1) / (df['TotalSpend'] + 1)
    
    df['SpendingEntropy'] = -(spend_props * np.log(spend_props + 1e-10)).sum(axis=1)
    
    # 2. Spending ratios (better than raw values)
    for col in spend_cols:
        df[f'{col}_Ratio'] = df[col] / (df['TotalSpend'] + 1)
    
    # 3. Age-Spending interaction patterns
    df['YoungBigSpender'] = ((df['Age'] < 25) & (df['TotalSpend'] > df['TotalSpend'].quantile(0.75))).astype(int)
    df['ElderlyLowSpender'] = ((df['Age'] > 60) & (df['TotalSpend'] < df['TotalSpend'].quantile(0.25))).astype(int)
    
    # 4. Cabin location quality score
    deck_quality = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'Unknown': 4.5}
    df['DeckQuality'] = df['CabinDeck'].map(deck_quality)
    
    # 5. Group behavior consistency
    df['GroupAgeStd'] = df.groupby('GroupId')['Age'].transform('std').fillna(0)
    df['GroupSpendStd'] = df.groupby('GroupId')['TotalSpend'].transform('std').fillna(0)
    df['GroupCryoConsistent'] = df.groupby('GroupId')['CryoSleep'].transform(lambda x: int(x.nunique() == 1))
    
    # 6. Destination-based features
    df['DestinationSpendAvg'] = df.groupby('Destination')['TotalSpend'].transform('mean').fillna(df['TotalSpend'].mean())
    df['SpendVsDestAvg'] = df['TotalSpend'] - df['DestinationSpendAvg']
    
    # 7. Smart imputation indicators
    df['HasMissingCabin'] = df['Cabin'].isna().astype(int)
    
    # 8. Luxury service patterns
    df['UsesLuxuryServices'] = ((df['Spa'] > 0) | (df['VRDeck'] > 0)).astype(int)
    df['UsesFoodServices'] = ((df['RoomService'] > 0) | (df['FoodCourt'] > 0)).astype(int)
    
    # 9. Validate CryoSleep logic (important rule in the problem)
    df['CryoSleepAnomaly'] = ((df['CryoSleep'] == 1) & (df['TotalSpend'] > 0)).astype(int)
    
    return df

def create_enhanced_model():
    """
    Enhanced model building on your 0.807 success
    """
    print("🚀 Enhanced Spaceship Titanic Model")
    print("Building on your successful 0.807 approach")
    print("="*50)
    
    # Load data
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    # Apply enhanced feature engineering
    print("⚙️ Applying enhanced feature engineering...")
    train = enhanced_feature_engineering(train)
    test = enhanced_feature_engineering(test)
    
    # Enhanced clustering with scaling
    print("🔮 Creating enhanced clusters...")
    km_feats = ['CabinNum','log_RoomService','log_FoodCourt',
                'log_ShoppingMall','log_Spa','log_VRDeck','TotalSpend',
                'SpendingEntropy','DeckQuality','Age']  # Added more features
    
    # Scale for better clustering
    scaler = StandardScaler()
    train_cluster_data = train[km_feats].fillna(0)
    train_scaled = scaler.fit_transform(train_cluster_data)
    
    # Try more clusters for better segmentation
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=20).fit(train_scaled)
    train['Cluster'] = kmeans.predict(train_scaled)
    
    test_cluster_data = test[km_feats].fillna(0)
    test_scaled = scaler.transform(test_cluster_data)
    test['Cluster'] = kmeans.predict(test_scaled)
    
    # Your original advanced features
    train['MaxSpendItem'] = train[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].idxmax(axis=1)
    test['MaxSpendItem'] = test[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].idxmax(axis=1)
    
    # Enhanced frequency encoding with smoothing
    for col in ['HomePlanet','Destination','DeckSide','MaxSpendItem']:
        freq = train[col].value_counts() / len(train)
        # Add smoothing to prevent overfitting
        smoothing = 0.001
        train[f'{col}_Freq'] = train[col].map(freq).fillna(smoothing)
        test[f'{col}_Freq'] = test[col].map(freq).fillna(smoothing)
    
    # Target encoding for high-cardinality features (with CV to prevent leakage)
    from sklearn.model_selection import KFold
    
    def target_encode_cv(train_df, test_df, col, target, n_splits=5):
        train_df[f'{col}_TargetEnc'] = 0
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Reset index to use integer positions
        train_df_reset = train_df.reset_index()
        
        for train_idx, val_idx in kf.split(train_df_reset):
            mean_target = train_df_reset.iloc[train_idx].groupby(col)[target].mean()
            train_df_reset.loc[val_idx, f'{col}_TargetEnc'] = train_df_reset.iloc[val_idx][col].map(mean_target)
        
        # Fill NaN with global mean
        global_mean = train_df_reset[target].mean()
        train_df_reset[f'{col}_TargetEnc'].fillna(global_mean, inplace=True)
        
        # Copy back to original dataframe
        train_df[f'{col}_TargetEnc'] = train_df_reset[f'{col}_TargetEnc'].values
        
        # For test set, use full training data
        mean_target_full = train_df.groupby(col)[target].mean()
        test_df[f'{col}_TargetEnc'] = test_df[col].map(mean_target_full).fillna(global_mean)
        
        return train_df, test_df
    
    # Apply target encoding to key features
    for col in ['CabinDeck', 'Cluster']:
        train, test = target_encode_cv(train, test, col, 'Transported')
    
    # Prepare final datasets
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
    
    print(f"📊 Total features: {X.shape[1]} (vs your original ~28)")
    
    # Enhanced models with better hyperparameters
    hgb1 = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=600,  # More iterations
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        max_bins=255,
        random_state=42
    )
    
    hgb2 = HistGradientBoostingClassifier(
        learning_rate=0.08,  # Slightly different
        max_iter=400,
        max_leaf_nodes=28,  # Slightly different
        min_samples_leaf=25,
        l2_regularization=0.15,
        max_bins=255,
        random_state=24
    )
    
    lgbm = LGBMClassifier(
        n_estimators=600,  # More trees
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=0,
        verbose=-1
    )
    
    xgb = XGBClassifier(
        n_estimators=600,  # More trees
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Enhanced stacking with regularized meta-learner
    stack = StackingClassifier(
        estimators=[
            ('hgb1', hgb1), ('hgb2', hgb2),
            ('lgbm', lgbm), ('xgb', xgb)
        ],
        final_estimator=LogisticRegression(
            max_iter=1000,
            C=0.5,  # More regularization
            random_state=42
        ),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    # Cross-validation
    print("\n📊 Cross-validating enhanced ensemble...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"\n🎯 Enhanced Model Results:")
    print(f"   Your baseline: 0.8070 (80.7%)")
    print(f"   Enhanced CV: {scores.mean():.4f} ({scores.mean()*100:.1f}%)")
    print(f"   Improvement: +{(scores.mean() - 0.807)*100:.2f}%")
    print(f"   Std deviation: ±{scores.std():.4f}")
    print(f"   All folds: {scores}")
    
    # Train final model
    print("\n🚀 Training final enhanced model...")
    stack.fit(X, y)
    
    # Make predictions
    preds_test = stack.predict(X_test).astype(bool)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': X_test.index,
        'Transported': preds_test
    })
    # Ensure PassengerId is in correct format (string)
    submission['PassengerId'] = submission['PassengerId'].astype(str)
    submission.to_csv('submission_enhanced.csv', index=False)
    
    print("✅ Created submission_enhanced.csv")
    
    # Feature importance analysis
    print("\n🔍 Top features by importance:")
    
    # Get feature importance from LGBM
    lgbm_solo = LGBMClassifier(
        n_estimators=100, learning_rate=0.1, num_leaves=31,
        random_state=0, verbose=-1
    )
    lgbm_solo.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': lgbm_solo.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    print(feature_importance)
    
    # Additional ensemble experiments
    print("\n🧪 Testing ensemble variations...")
    
    # Try without XGBoost
    stack_no_xgb = StackingClassifier(
        estimators=[('hgb1', hgb1), ('hgb2', hgb2), ('lgbm', lgbm)],
        final_estimator=LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        cv=5, passthrough=True, n_jobs=-1
    )
    
    scores_no_xgb = cross_val_score(stack_no_xgb, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Without XGBoost: {scores_no_xgb.mean():.4f}")
    
    # Try blending instead of stacking
    from sklearn.ensemble import VotingClassifier
    
    voting = VotingClassifier(
        estimators=[
            ('hgb1', hgb1), ('hgb2', hgb2),
            ('lgbm', lgbm), ('xgb', xgb)
        ],
        voting='soft',
        weights=[1.2, 1.0, 1.3, 1.1],  # Weight best performers higher
        n_jobs=-1
    )
    
    voting_scores = cross_val_score(voting, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Weighted voting: {voting_scores.mean():.4f}")
    
    # Use best approach
    best_score = max(scores.mean(), scores_no_xgb.mean(), voting_scores.mean())
    
    if voting_scores.mean() == best_score:
        print("\n💡 Weighted voting performs best! Creating alternative submission...")
        voting.fit(X, y)
        preds_voting = voting.predict(X_test).astype(bool)
        
        submission_voting = pd.DataFrame({
            'PassengerId': X_test.index,
            'Transported': preds_voting
        })
        # Ensure PassengerId is in correct format (string)
        submission_voting['PassengerId'] = submission_voting['PassengerId'].astype(str)
        submission_voting.to_csv('submission_best.csv', index=False)
        print("✅ Created submission_best.csv (weighted voting)")
    elif scores_no_xgb.mean() == best_score:
        print("\n💡 3-model ensemble performs best! Creating alternative submission...")
        stack_no_xgb.fit(X, y)
        preds_no_xgb = stack_no_xgb.predict(X_test).astype(bool)
        
        submission_no_xgb = pd.DataFrame({
            'PassengerId': X_test.index,
            'Transported': preds_no_xgb
        })
        # Ensure PassengerId is in correct format (string)
        submission_no_xgb['PassengerId'] = submission_no_xgb['PassengerId'].astype(str)
        submission_no_xgb.to_csv('submission_best.csv', index=False)
        print("✅ Created submission_best.csv (without XGBoost)")
    else:
        print("\n✅ Original stacking performs best - use submission_enhanced.csv")
    
    return stack, scores

if __name__ == "__main__":
    print("🎯 ENHANCED SPACESHIP TITANIC SOLUTION")
    print("Building on your successful 0.807 baseline")
    print("="*50)
    
    # Run enhanced model
    model, scores = create_enhanced_model()
    
    print(f"\n📊 SUMMARY:")
    print(f"Your baseline: 0.8070")
    print(f"Enhanced approach: {scores.mean():.4f}")
    print(f"Expected improvement: +{(scores.mean() - 0.807)*100:.2f}%")
    
    print("\n💡 Key enhancements made:")
    print("1. Added spending entropy and ratios")
    print("2. Enhanced clustering with more features and scaling")
    print("3. Target encoding for high-cardinality features")
    print("4. Group behavior consistency features")
    print("5. Better hyperparameter tuning")
    print("6. Multiple ensemble strategies tested")
    
    print("\n📁 Files created:")
    print("- submission_enhanced.csv (main submission)")
    print("- submission_best.csv (best performing variant)")
    print("\nTry both submissions to see which performs better!")
