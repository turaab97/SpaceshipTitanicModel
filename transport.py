import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, ExtraTreesClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

def ultra_advanced_filling(df):
    """
    Enhanced data filling with logical constraints from notebook analysis
    """
    df = df.copy()
    print("🔬 Ultra-Advanced Multi-Algorithm Data Filling")
    
    # FIRST: Define all variables we'll need
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Create basic features
    df['GroupId'] = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('GroupId')['HomePlanet'].transform('size')
    
    # Parse cabin information
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['CabinDeck'] = cabin_split[0] if len(cabin_split.columns) > 0 else None
    df['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce') if len(cabin_split.columns) > 1 else None
    df['CabinSide'] = cabin_split[2] if len(cabin_split.columns) > 2 else None
    
    # Create total expenses (handling NaN properly)
    df['TotalExpenses'] = df[spend_cols].fillna(0).sum(axis=1)
    
    # KEY INSIGHT FROM NOTEBOOK: Logical CryoSleep filling
    # If someone has 0 expenses and CryoSleep is missing → they're likely in CryoSleep
    # If someone has expenses and CryoSleep is missing → they're likely NOT in CryoSleep
    cryo_missing = df['CryoSleep'].isna()
    df.loc[cryo_missing & (df['TotalExpenses'] == 0), 'CryoSleep'] = True
    df.loc[cryo_missing & (df['TotalExpenses'] > 0), 'CryoSleep'] = False
    
    # Fill any remaining CryoSleep with mode
    if df['CryoSleep'].isna().any():
        df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mode()[0])
    
    # KEY INSIGHT: CryoSleep passengers should have 0 expenses
    for col in spend_cols:
        if df[col].isna().any():
            # CryoSleep passengers get 0 expenses
            cryo_mask = (df['CryoSleep'] == True) & (df[col].isna())
            df.loc[cryo_mask, col] = 0
            
            # Awake passengers get mean of awake passengers
            awake_mask = (df['CryoSleep'] == False) & (df[col].isna())
            if awake_mask.any():
                awake_mean = df[df['CryoSleep'] == False][col].mean()
                awake_mean = 0 if pd.isna(awake_mean) else awake_mean
                df.loc[awake_mask, col] = awake_mean
    
    # Recalculate total expenses
    df['TotalExpenses'] = df[spend_cols].sum(axis=1)
    
    # Fill other missing values with simple but effective methods
    # Age
    if df['Age'].isna().any():
        df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # VIP
    if df['VIP'].isna().any():
        df['VIP'] = df['VIP'].fillna(False)
    
    # Categorical variables
    for col in ['HomePlanet', 'Destination']:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            df[col] = df[col].fillna(fill_val)
    
    # Cabin components
    for col in ['CabinDeck', 'CabinSide']:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            df[col] = df[col].fillna(fill_val)
    
    if 'CabinNum' in df.columns and df['CabinNum'].isna().any():
        df['CabinNum'] = df['CabinNum'].fillna(df['CabinNum'].median())
    
    print("✅ Enhanced logical filling completed")
    return df

def maximum_feature_engineering(df):
    """
    Enhanced feature engineering incorporating notebook insights
    """
    df = df.copy()
    print("⚙️ Maximum Feature Engineering for 0.813 Target")
    
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Core group features
    df['GroupId'] = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('GroupId')['HomePlanet'].transform('size')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    
    # Enhanced cabin features (KEY INSIGHT FROM NOTEBOOK)
    df['DeckSide'] = df['CabinDeck'].astype(str) + '_' + df['CabinSide'].astype(str)
    df['CabinNumOdd'] = (df['CabinNum'] % 2).astype(int)
    df['CabinNumBin'] = pd.qcut(df['CabinNum'], q=20, labels=False, duplicates='drop')
    df['CabinQuadrant'] = pd.cut(df['CabinNum'], bins=4, labels=['Q1','Q2','Q3','Q4'])
    
    # Keep individual cabin components as separate features (notebook showed these are important)
    df['CabinDeck_encoded'] = pd.Categorical(df['CabinDeck']).codes
    df['CabinSide_encoded'] = pd.Categorical(df['CabinSide']).codes
    
    # Enhanced spending features
    if 'TotalExpenses' in df.columns:
        df['TotalSpend'] = df['TotalExpenses']
    else:
        df['TotalSpend'] = df[spend_cols].sum(axis=1)
    
    # Basic transformations
    for c in spend_cols:
        df[f'log_{c}'] = np.log1p(df[c])
        df[f'sqrt_{c}'] = np.sqrt(df[c])
        df[f'has_{c}'] = (df[c] > 0).astype(int)
        df[f'high_{c}'] = (df[c] > df[c].quantile(0.75)).astype(int)
    
    # Spending aggregates
    df['SpendPerPerson'] = df['TotalSpend'] / (df['GroupSize'] + 1)
    df['SpendVariance'] = df[spend_cols].var(axis=1)
    df['SpendStd'] = df[spend_cols].std(axis=1)
    df['SpendSkew'] = df[spend_cols].skew(axis=1)
    df['MaxSpend'] = df[spend_cols].max(axis=1)
    df['MinSpend'] = df[spend_cols].min(axis=1)
    df['SpendRange'] = df['MaxSpend'] - df['MinSpend']
    df['SpendMedian'] = df[spend_cols].median(axis=1)
    df['NonZeroSpends'] = (df[spend_cols] > 0).sum(axis=1)
    df['SpendConcentration'] = df['MaxSpend'] / (df['TotalSpend'] + 1)
    
    # KEY INSIGHTS FROM NOTEBOOK: Spending pattern features
    df['HasAnyExpenses'] = (df['TotalSpend'] > 0).astype(int)
    df['HighSpender'] = (df['TotalSpend'] > df['TotalSpend'].quantile(0.8)).astype(int)
    df['LowSpender'] = (df['TotalSpend'] < df['TotalSpend'].quantile(0.2)).astype(int)
    
    # Age features
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,25,35,50,65,np.inf],
                         labels=['Child','Teen','YoungAdult','Adult','MiddleAge','Senior','Elder'])
    df['IsChild'] = (df['Age'] < 18).astype(int)
    df['IsTeen'] = ((df['Age'] >= 13) & (df['Age'] < 20)).astype(int)
    df['IsYoungAdult'] = ((df['Age'] >= 18) & (df['Age'] < 30)).astype(int)
    df['IsMiddleAge'] = ((df['Age'] >= 35) & (df['Age'] < 55)).astype(int)
    df['IsSenior'] = (df['Age'] >= 60).astype(int)
    df['AgeSpendRatio'] = df['Age'] / (df['TotalSpend'] + 1)
    df['AgeGroupRatio'] = df['Age'] / (df['GroupSize'] + 1)
    
    # Boolean features
    for col in ['CryoSleep','VIP']:
        df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0}).astype(int)
    
    # Enhanced interaction features (incorporating notebook logic)
    df['CryoSpendConsistency'] = ((df['CryoSleep'] == 1) & (df['TotalSpend'] == 0)).astype(int)
    df['CryoSpendAnomaly'] = ((df['CryoSleep'] == 1) & (df['TotalSpend'] > 0)).astype(int)
    
    # VIP spending patterns (more granular)
    df['VIPHighSpender'] = ((df['VIP'] == 1) & (df['TotalSpend'] > df['TotalSpend'].quantile(0.8))).astype(int)
    df['VIPLowSpender'] = ((df['VIP'] == 1) & (df['TotalSpend'] < df['TotalSpend'].quantile(0.5))).astype(int)
    df['VIPNoSpend'] = ((df['VIP'] == 1) & (df['TotalSpend'] == 0)).astype(int)
    df['VIPCryoCombo'] = ((df['VIP'] == 1) & (df['CryoSleep'] == 1)).astype(int)
    
    # Family and group patterns
    df['FamilyWithKids'] = ((df['GroupSize'] > 1) & (df['Age'] < 18)).astype(int)
    df['FamilyWithTeens'] = ((df['GroupSize'] > 1) & (df['Age'] >= 13) & (df['Age'] < 20)).astype(int)
    df['LargeFamily'] = (df['GroupSize'] >= 4).astype(int)
    df['CoupleTrip'] = (df['GroupSize'] == 2).astype(int)
    df['SoloTraveler'] = (df['GroupSize'] == 1).astype(int)
    df['GroupTrip'] = (df['GroupSize'] >= 3).astype(int)
    
    # Advanced group features
    group_stats = df.groupby('GroupId').agg({
        'TotalSpend': ['mean', 'std', 'min', 'max', 'sum'],
        'Age': ['mean', 'std', 'min', 'max'],
        'CryoSleep': ['mean', 'sum'],
        'VIP': ['mean', 'sum'],
        'NonZeroSpends': ['mean', 'std']
    }).reset_index()
    
    group_stats.columns = ['GroupId'] + [f'Group_{col[0]}_{col[1]}' for col in group_stats.columns[1:]]
    group_stats = group_stats.fillna(0)
    df = df.merge(group_stats, on='GroupId', how='left')
    
    # Group coherence features
    df['SpendVsGroupMean'] = df['TotalSpend'] - df['Group_TotalSpend_mean']
    df['SpendVsGroupMax'] = df['TotalSpend'] - df['Group_TotalSpend_max']
    df['IsGroupTopSpender'] = (df['TotalSpend'] == df['Group_TotalSpend_max']).astype(int)
    df['IsGroupBottomSpender'] = (df['TotalSpend'] == df['Group_TotalSpend_min']).astype(int)
    df['AgeVsGroupMean'] = df['Age'] - df['Group_Age_mean']
    df['IsOldestInGroup'] = (df['Age'] == df['Group_Age_max']).astype(int)
    df['IsYoungestInGroup'] = (df['Age'] == df['Group_Age_min']).astype(int)
    
    # Planet and destination patterns
    df['HomeDest'] = df['HomePlanet'] + '_' + df['Destination']
    df['IsRoundTrip'] = (df['HomePlanet'] == df['Destination']).astype(int)
    
    return df

def main_ensemble_model():
    """
    Main ensemble model with enhanced preprocessing
    """
    print("🚀 SPACESHIP TITANIC ENSEMBLE MODEL")
    print("="*50)
    
    # Load data WITHOUT setting PassengerId as index
    print("📂 Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    # Store original PassengerId for final submission
    test_passenger_ids = test['PassengerId'].copy()
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Sample PassengerId format: {test_passenger_ids.iloc[0]}")
    
    # Data preprocessing
    print("\n🔬 Step 1: Ultra-Advanced Data Filling...")
    train = ultra_advanced_filling(train)
    test = ultra_advanced_filling(test)
    
    print("\n⚙️ Step 2: Maximum Feature Engineering...")
    train = maximum_feature_engineering(train)
    test = maximum_feature_engineering(test)
    
    # Additional clustering features
    print("\n📊 Step 3: Advanced Clustering Features...")
    
    # Spending behavior clustering
    spend_features = [col for col in train.columns if 'log_' in col or 'sqrt_' in col]
    spend_features.extend(['TotalSpend', 'SpendVariance', 'NonZeroSpends'])
    
    scaler = RobustScaler()
    spend_scaled = scaler.fit_transform(train[spend_features].fillna(0))
    
    for k in [5, 8, 12]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        train[f'SpendCluster_{k}'] = kmeans.fit_predict(spend_scaled)
        test[f'SpendCluster_{k}'] = kmeans.predict(scaler.transform(test[spend_features].fillna(0)))
    
    # Add max spend item
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    train['MaxSpendItem'] = train[spend_cols].idxmax(axis=1)
    test['MaxSpendItem'] = test[spend_cols].idxmax(axis=1)
    
    # Frequency encodings
    freq_cols = ['HomePlanet','Destination','DeckSide','HomeDest','MaxSpendItem']
    for col in freq_cols:
        if col in train.columns:
            freq = train[col].value_counts() / len(train)
            train[f'{col}_Freq'] = train[col].map(freq).fillna(0)
            test[f'{col}_Freq'] = test[col].map(freq).fillna(0)
    
    # Target encoding with proper cross-validation
    target_cols = ['CabinDeck', 'AgeBin', 'HomeDest']
    for col in target_cols:
        if col in train.columns:
            # Convert categorical to string to avoid dtype issues
            if hasattr(train[col], 'cat'):
                train[col] = train[col].astype(str)
                test[col] = test[col].astype(str)
            
            cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train[f'{col}_TargetEnc'] = 0.0
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_folds.split(train, train['Transported'])):
                train_fold = train.iloc[train_idx]
                target_mean = train_fold.groupby(col)['Transported'].mean()
                global_mean = train['Transported'].mean()
                
                val_data = train.iloc[val_idx]
                target_enc_values = val_data[col].map(target_mean).fillna(global_mean)
                train.iloc[val_idx, train.columns.get_loc(f'{col}_TargetEnc')] = target_enc_values
            
            # For test set
            target_mean = train.groupby(col)['Transported'].mean()
            global_mean = train['Transported'].mean()
            test[f'{col}_TargetEnc'] = test[col].map(target_mean).fillna(global_mean)
    
    # Prepare final dataset
    y = train['Transported'].astype(int)
    drop_cols = ['Transported','Name','Cabin','GroupId','PassengerId'] + freq_cols + target_cols
    X = train.drop(columns=drop_cols)
    
    test_drop_cols = [col for col in drop_cols if col in test.columns and col != 'Transported']
    X_test = test.drop(columns=test_drop_cols)
    
    # Handle remaining categoricals
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
        X_test[c] = X_test[c].astype('category').cat.codes
    
    print(f"📊 Final feature count: {X.shape[1]}")
    
    # Feature selection
    print("\n🎯 Step 4: Feature Selection...")
    mi_selector = SelectKBest(mutual_info_classif, k=min(80, X.shape[1]))
    X_selected = mi_selector.fit_transform(X, y)
    X_test_selected = mi_selector.transform(X_test)
    
    print(f"   Selected features: {X_selected.shape[1]}")
    
    # Ultimate model ensemble
    print("\n🚀 Step 5: Training Ensemble Model...")
    
    # Base models
    base_models = [
        ('hgb', HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=1000, max_leaf_nodes=31,
            min_samples_leaf=25, l2_regularization=0.1, random_state=42)),
        
        ('lgbm', LGBMClassifier(
            n_estimators=1000, learning_rate=0.03, num_leaves=20,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1)),
        
        ('xgb', XGBClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=4,
            min_child_weight=6, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, eval_metric='logloss')),
        
        ('cat', CatBoostClassifier(
            iterations=800, learning_rate=0.03, depth=4,
            l2_leaf_reg=10, random_seed=42, verbose=False)),
        
        ('et', ExtraTreesClassifier(
            n_estimators=500, max_depth=10, min_samples_split=8,
            min_samples_leaf=4, random_state=42, n_jobs=-1))
    ]
    
    # Stacking ensemble
    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(C=0.5, max_iter=3000, random_state=42),
        cv=8, passthrough=True, n_jobs=-1
    )
    
    # Cross-validation evaluation
    print("\n🧪 Step 6: Model Evaluation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stack, X_selected, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"\n🏆 MODEL PERFORMANCE:")
    print(f"   Cross-validation score: {scores.mean():.5f} ± {scores.std():.5f}")
    print(f"   Individual fold scores: {[f'{s:.4f}' for s in scores]}")
    
    # Train final model and make predictions
    print("\n🚀 Step 7: Final Training and Prediction...")
    stack.fit(X_selected, y)
    predictions = stack.predict(X_test_selected)
    
    # Create submission in the exact format required
    submission = pd.DataFrame({
        'PassengerId': test_passenger_ids,  # Use original PassengerId format
        'Transported': predictions.astype(bool)  # Convert to boolean as in sample
    })
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    
    print(f"\n✅ SUBMISSION CREATED!")
    print(f"📁 File: submission.csv")
    print(f"📊 Shape: {submission.shape}")
    print(f"🎯 CV Score: {scores.mean():.5f}")
    print(f"\nFirst 5 predictions:")
    print(submission.head())
    print(f"\nPrediction distribution:")
    print(submission['Transported'].value_counts())
    
    return stack, scores.mean()

if __name__ == "__main__":
    model, score = main_ensemble_model()
    print(f"\n🎉 Model training completed with CV score: {score:.5f}")
    print("📧 Ready for submission!")
