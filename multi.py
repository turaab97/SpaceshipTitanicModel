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
    Multi-algorithm data filling for maximum accuracy
    """
    df = df.copy()
    print("🔬 Ultra-Advanced Multi-Algorithm Data Filling")
    
    # Create comprehensive features for filling
    df['GroupId'] = df.index.str.split('_').str[0]
    df['GroupSize'] = df.groupby('GroupId')['HomePlanet'].transform('size')
    
    cabin = df['Cabin'].str.split('/', expand=True)
    df['CabinDeck'] = cabin[0]
    df['CabinNum'] = pd.to_numeric(cabin[1], errors='coerce')
    df['CabinSide'] = cabin[2]
    
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    
    # Multi-level clustering for different aspects
    
    # 1. Demographic clustering
    demo_features = []
    df['Age_temp'] = df['Age'].fillna(df['Age'].median())
    demo_features.append('Age_temp')
    demo_features.append('GroupSize')
    
    for col in ['HomePlanet', 'Destination']:
        df[f'{col}_encoded'] = df[col].astype('category').cat.codes
        demo_features.append(f'{col}_encoded')
    
    # 2. Behavioral clustering (spending)
    behavior_features = []
    for col in spend_cols:
        df[f'{col}_temp'] = df[col].fillna(0)
        df[f'{col}_log'] = np.log1p(df[f'{col}_temp'])
        behavior_features.extend([f'{col}_temp', f'{col}_log'])
    
    df['TotalSpend_temp'] = df[[f'{col}_temp' for col in spend_cols]].sum(axis=1)
    behavior_features.append('TotalSpend_temp')
    
    # 3. Location clustering
    location_features = []
    df['CabinNum_temp'] = df['CabinNum'].fillna(df['CabinNum'].median())
    location_features.append('CabinNum_temp')
    
    for col in ['CabinDeck', 'CabinSide']:
        df[f'{col}_loc_encoded'] = df[col].astype('category').cat.codes
        location_features.append(f'{col}_loc_encoded')
    
    # Perform multiple clustering algorithms
    scaler = RobustScaler()
    
    # KMeans clusters
    for name, features, n_clusters in [
        ('demo', demo_features, 8),
        ('behavior', behavior_features, 10),
        ('location', location_features, 6)
    ]:
        X_cluster = scaler.fit_transform(df[features].fillna(0))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        df[f'Cluster_{name}'] = kmeans.fit_predict(X_cluster)
    
    # Hierarchical clustering for additional structure
    all_features = demo_features + behavior_features + location_features
    X_all = scaler.fit_transform(df[all_features].fillna(0))
    
    # Agglomerative clustering
    agg_cluster = AgglomerativeClustering(n_clusters=12, linkage='ward')
    df['Cluster_hierarchical'] = agg_cluster.fit_predict(X_all)
    
    print(f"📊 Created multiple clustering views for intelligent filling")
    
    # Advanced filling using ensemble of clusters
    
    # Age filling with multiple cluster consensus
    if df['Age'].isna().any():
        missing_age = df['Age'].isna()
        cluster_cols = ['Cluster_demo', 'Cluster_behavior', 'Cluster_hierarchical']
        
        for idx in df[missing_age].index:
            age_predictions = []
            
            for cluster_col in cluster_cols:
                cluster_val = df.loc[idx, cluster_col]
                cluster_ages = df[df[cluster_col] == cluster_val]['Age'].dropna()
                
                if len(cluster_ages) > 0:
                    age_predictions.append(cluster_ages.median())
            
            if age_predictions:
                df.loc[idx, 'Age'] = np.median(age_predictions)
            else:
                df.loc[idx, 'Age'] = df['Age'].median()
        
        print(f"✅ Multi-cluster Age filling completed")
    
    # Spending filling with advanced logic
    for col in spend_cols:
        if df[col].isna().any():
            missing_spend = df[col].isna()
            
            for idx in df[missing_spend].index:
                # Check CryoSleep first
                cryo_status = df.loc[idx, 'CryoSleep']
                if pd.notna(cryo_status) and cryo_status in [True, 'True', 1]:
                    df.loc[idx, col] = 0
                    continue
                
                # Use behavior cluster
                behavior_cluster = df.loc[idx, 'Cluster_behavior']
                behavior_data = df[df['Cluster_behavior'] == behavior_cluster]
                
                if not behavior_data[col].empty:
                    vip_status = df.loc[idx, 'VIP']
                    if pd.notna(vip_status) and vip_status in [True, 'True', 1]:
                        # VIP tends to spend more
                        spend_val = behavior_data[col].quantile(0.7)
                    else:
                        spend_val = behavior_data[col].median()
                    
                    if pd.notna(spend_val):
                        df.loc[idx, col] = max(0, spend_val)
                    else:
                        df.loc[idx, col] = 0
                else:
                    df.loc[idx, col] = 0
            
            print(f"✅ Advanced {col} filling completed")
    
    # Boolean filling with probability-based approach
    for col in ['CryoSleep', 'VIP']:
        if df[col].isna().any():
            missing_bool = df[col].isna()
            
            for idx in df[missing_bool].index:
                # Use multiple clusters to estimate probability
                prob_true = 0
                count = 0
                
                for cluster_col in ['Cluster_demo', 'Cluster_behavior', 'Cluster_hierarchical']:
                    cluster_val = df.loc[idx, cluster_col]
                    cluster_data = df[df[cluster_col] == cluster_val]
                    
                    if not cluster_data[col].empty:
                        cluster_prob = cluster_data[col].map({True:1, False:0, 'True':1, 'False':0}).mean()
                        if pd.notna(cluster_prob):
                            prob_true += cluster_prob
                            count += 1
                
                if count > 0:
                    final_prob = prob_true / count
                    df.loc[idx, col] = final_prob > 0.5
                else:
                    df.loc[idx, col] = False
            
            print(f"✅ Probability-based {col} filling completed")
    
    # Categorical filling with hierarchy
    for col in ['HomePlanet', 'Destination']:
        if df[col].isna().any():
            missing_cat = df[col].isna()
            
            for idx in df[missing_cat].index:
                # Try location cluster first
                location_cluster = df.loc[idx, 'Cluster_location']
                location_data = df[df['Cluster_location'] == location_cluster]
                
                if not location_data[col].empty:
                    mode_val = location_data[col].mode()
                    if len(mode_val) > 0:
                        df.loc[idx, col] = mode_val[0]
                        continue
                
                # Fallback to demo cluster
                demo_cluster = df.loc[idx, 'Cluster_demo']
                demo_data = df[df['Cluster_demo'] == demo_cluster]
                
                if not demo_data[col].empty:
                    mode_val = demo_data[col].mode()
                    if len(mode_val) > 0:
                        df.loc[idx, col] = mode_val[0]
                        continue
                
                # Final fallback
                df.loc[idx, col] = df[col].mode()[0]
            
            print(f"✅ Hierarchical {col} filling completed")
    
    # Cabin filling
    for col in ['CabinDeck', 'CabinSide']:
        if df[col].isna().any():
            missing_cabin = df[col].isna()
            
            for idx in df[missing_cabin].index:
                location_cluster = df.loc[idx, 'Cluster_location']
                location_data = df[df['Cluster_location'] == location_cluster]
                
                if not location_data[col].empty:
                    mode_val = location_data[col].mode()
                    df.loc[idx, col] = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                else:
                    df.loc[idx, col] = 'Unknown'
    
    if df['CabinNum'].isna().any():
        missing_cabin_num = df['CabinNum'].isna()
        for idx in df[missing_cabin_num].index:
            location_cluster = df.loc[idx, 'Cluster_location']
            location_data = df[df['Cluster_location'] == location_cluster]
            
            if not location_data['CabinNum'].empty:
                df.loc[idx, 'CabinNum'] = location_data['CabinNum'].median()
            else:
                df.loc[idx, 'CabinNum'] = df['CabinNum'].median()
    
    # Clean up temporary columns
    temp_cols = [col for col in df.columns if '_temp' in col or '_encoded' in col or '_log' in col or 'Cluster_' in col]
    df = df.drop(columns=temp_cols)
    
    return df

def maximum_feature_engineering(df):
    """
    Maximum feature engineering for 0.813 target
    """
    df = df.copy()
    print("⚙️ Maximum Feature Engineering for 0.813 Target")
    
    # Core group features
    df['GroupId'] = df.index.str.split('_').str[0]
    df['GroupSize'] = df.groupby('GroupId')['HomePlanet'].transform('size')
    df['IsAlone'] = (df['GroupSize']==1).astype(int)
    
    # Enhanced cabin features
    df['DeckSide'] = df['CabinDeck'] + '_' + df['CabinSide']
    df['CabinNumOdd'] = (df['CabinNum'] % 2).astype(int)
    df['CabinNumBin'] = pd.qcut(df['CabinNum'], q=20, labels=False, duplicates='drop')
    df['CabinQuadrant'] = pd.cut(df['CabinNum'], bins=4, labels=['Q1','Q2','Q3','Q4'])
    
    # Maximum spending features
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    
    # Basic transformations
    for c in spend_cols:
        df[f'log_{c}'] = np.log1p(df[c])
        df[f'sqrt_{c}'] = np.sqrt(df[c])
        df[f'has_{c}'] = (df[c] > 0).astype(int)
        df[f'high_{c}'] = (df[c] > df[c].quantile(0.75)).astype(int)
    
    # Spending aggregates
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
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
    
    # Maximum interaction features
    df['CryoSpendFlag'] = (df['CryoSleep'] == 1) & (df['TotalSpend'] == 0)
    df['CryoSpendFlag'] = df['CryoSpendFlag'].astype(int)
    
    df['CryoSpendAnomaly'] = (df['CryoSleep'] == 1) & (df['TotalSpend'] > 0)
    df['CryoSpendAnomaly'] = df['CryoSpendAnomaly'].astype(int)
    
    df['VIPHighSpender'] = (df['VIP'] == 1) & (df['TotalSpend'] > df['TotalSpend'].quantile(0.8))
    df['VIPHighSpender'] = df['VIPHighSpender'].astype(int)
    
    df['VIPNoSpend'] = (df['VIP'] == 1) & (df['TotalSpend'] == 0)
    df['VIPNoSpend'] = df['VIPNoSpend'].astype(int)
    
    df['VIPCryoCombo'] = (df['VIP'] == 1) & (df['CryoSleep'] == 1)
    df['VIPCryoCombo'] = df['VIPCryoCombo'].astype(int)
    
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

def ultimate_ensemble_0813():
    """
    Ultimate ensemble targeting 0.813
    """
    print("🚀 ULTIMATE ENSEMBLE - TARGET 0.813")
    print("Current: 0.80617 → Target: 0.813 (+0.00683 needed)")
    print("="*60)
    
    # Load data
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    # Ultra-advanced data filling
    print("🔬 Step 1: Ultra-Advanced Data Filling...")
    train = ultra_advanced_filling(train)
    test = ultra_advanced_filling(test)
    
    # Maximum feature engineering
    print("⚙️ Step 2: Maximum Feature Engineering...")
    train = maximum_feature_engineering(train)
    test = maximum_feature_engineering(test)
    
    # Multiple clustering for additional features
    print("📊 Step 3: Advanced Clustering Features...")
    
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
    
    # Target encoding (with proper CV) - Fix indexing issue
    target_cols = ['CabinDeck', 'AgeBin', 'HomeDest']
    for col in target_cols:
        if col in train.columns:
            # Convert categorical to string to avoid dtype issues
            if hasattr(train[col], 'cat'):
                train[col] = train[col].astype(str)
                test[col] = test[col].astype(str)
            
            cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train[f'{col}_TargetEnc'] = 0.0  # Initialize as float
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_folds.split(train, train['Transported'])):
                train_fold = train.iloc[train_idx]
                target_mean = train_fold.groupby(col)['Transported'].mean()
                global_mean = train['Transported'].mean()
                
                # Use iloc for proper integer indexing
                val_data = train.iloc[val_idx]
                target_enc_values = val_data[col].map(target_mean).fillna(global_mean)
                train.iloc[val_idx, train.columns.get_loc(f'{col}_TargetEnc')] = target_enc_values
            
            # For test set
            target_mean = train.groupby(col)['Transported'].mean()
            global_mean = train['Transported'].mean()
            test[f'{col}_TargetEnc'] = test[col].map(target_mean).fillna(global_mean)
    
    # Prepare final dataset
    y = train['Transported'].astype(int)
    drop_cols = ['Transported','Name','Cabin','GroupId'] + freq_cols + target_cols
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
    print("🎯 Step 4: Advanced Feature Selection...")
    
    # Mutual information selection
    mi_selector = SelectKBest(mutual_info_classif, k=min(80, X.shape[1]))
    X_mi = mi_selector.fit_transform(X, y)
    X_test_mi = mi_selector.transform(X_test)
    
    print(f"   Mutual Info features: {X_mi.shape[1]}")
    
    # Ultimate model ensemble
    print("🚀 Step 5: Ultimate Model Ensemble...")
    
    # Level 1: Diverse base models
    base_models = [
        ('hgb1', HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=1000, max_leaf_nodes=31,
            min_samples_leaf=25, l2_regularization=0.1, random_state=42)),
        
        ('hgb2', HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=600, max_leaf_nodes=25,
            min_samples_leaf=20, l2_regularization=0.08, random_state=123)),
        
        ('lgbm1', LGBMClassifier(
            n_estimators=1000, learning_rate=0.03, num_leaves=20,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1)),
        
        ('lgbm2', LGBMClassifier(
            n_estimators=800, learning_rate=0.04, num_leaves=25,
            min_child_samples=25, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.08, reg_lambda=0.08, random_state=999, verbose=-1)),
        
        ('xgb1', XGBClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=4,
            min_child_weight=6, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, eval_metric='logloss')),
        
        ('cat1', CatBoostClassifier(
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
        cv=10, passthrough=True, n_jobs=-1
    )
    
    # Test ensemble
    print("🧪 Step 6: Testing Ensemble...")
    cv_robust = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = cross_val_score(stack, X, y, cv=cv_robust, scoring='accuracy', n_jobs=-1)
    
    print(f"\n🏆 ENSEMBLE RESULTS:")
    print(f"   Target: 0.813")
    print(f"   Current best: 0.80617")
    print(f"   Ensemble CV: {scores.mean():.5f} ± {scores.std():.5f}")
    print(f"   vs target: {(scores.mean() - 0.813)*100:+.2f}%")
    print(f"   vs current: {(scores.mean() - 0.80617)*100:+.2f}%")
    print(f"   Individual folds: {[f'{s:.4f}' for s in scores]}")
    
    # Train final models
    print("🚀 Step 7: Training Final Models...")
    
    # Train main stack
    stack.fit(X, y)
    preds_stack = stack.predict(X_test).astype(bool)
    
    # Train on MI features
    stack_mi = StackingClassifier(
        estimators=base_models[:5],  # Use subset for MI features
        final_estimator=LogisticRegression(C=0.5, max_iter=3000, random_state=42),
        cv=8, passthrough=True, n_jobs=-1
    )
    stack_mi.fit(X_mi, y)
    preds_mi = stack_mi.predict(X_test_mi).astype(bool)
    
    # Simple probability averaging
    prob_models = [
        LGBMClassifier(n_estimators=1200, learning_rate=0.025, num_leaves=18, random_state=1111, verbose=-1),
        XGBClassifier(n_estimators=1000, learning_rate=0.025, max_depth=4, random_state=2222, eval_metric='logloss'),
        CatBoostClassifier(iterations=800, learning_rate=0.025, depth=4, random_seed=3333, verbose=False)
    ]
    
    prob_preds = []
    for model in prob_models:
        model.fit(X, y)
        probs = model.predict_proba(X_test)[:, 1]
        prob_preds.append(probs)
    
    avg_probs = np.mean(prob_preds, axis=0)
    preds_prob = (avg_probs > 0.5).astype(bool)
    
    # Create submissions
    submissions = [
        ('main_stack', preds_stack),
        ('mi_stack', preds_mi),
        ('prob_avg', preds_prob)
    ]
    
    for name, preds in submissions:
        submission = pd.DataFrame({
            'PassengerId': X_test.index.astype(str),
            'Transported': preds
        })
        submission.to_csv(f'submission_{name}_0813.csv', index=False)
        print(f"✅ Created submission_{name}_0813.csv")
    
    # Weighted ensemble of all approaches
    all_preds = np.array([preds_stack.astype(int), preds_mi.astype(int), preds_prob.astype(int)])
    ensemble_preds = (np.mean(all_preds, axis=0) > 0.5).astype(bool)
    
    submission_ensemble = pd.DataFrame({
        'PassengerId': X_test.index.astype(str),
        'Transported': ensemble_preds
    })
    submission_ensemble.to_csv('submission_ensemble_0813.csv', index=False)
    print("✅ Created submission_ensemble_0813.csv")
    
    print(f"\n📊 SUMMARY - TARGET 0.813:")
    print(f"Current baseline: 0.80617")
    print(f"Gap to close: +{(0.813 - 0.80617)*100:.2f}%")
    
    if scores.mean() > 0.813:
        print(f"🎉 SUCCESS! Exceeded target by +{(scores.mean() - 0.813)*100:.2f}%")
    elif scores.mean() > 0.80617:
        print(f"📈 PROGRESS! Improved by +{(scores.mean() - 0.80617)*100:.2f}%")
        print(f"   Still need +{(0.813 - scores.mean())*100:.2f}% more")
    else:
        print(f"⚠️ Need different approach. Current gap: {(0.813 - scores.mean())*100:+.2f}%")
    
    print(f"\n📁 Submissions created for 0.813 target:")
    print(f"   🎯 submission_main_stack_0813.csv")
    print(f"   🔬 submission_mi_stack_0813.csv")
    print(f"   🎲 submission_prob_avg_0813.csv")
    print(f"   🏆 submission_ensemble_0813.csv")
    print(f"\n💡 Strategy: Try all submissions - ensemble often performs best!")
    
    return stack, scores.mean()

def additional_advanced_strategies():
    """
    Additional advanced strategies if main approach doesn't reach 0.813
    """
    print("\n🧪 ADDITIONAL ADVANCED STRATEGIES")
    print("="*40)
    
    # Load and prepare data quickly
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    train = ultra_advanced_filling(train)
    test = ultra_advanced_filling(test)
    train = maximum_feature_engineering(train)
    test = maximum_feature_engineering(test)
    
    # Quick preprocessing
    y = train['Transported'].astype(int)
    drop_cols = ['Transported','Name','Cabin','GroupId','HomePlanet','Destination']
    X = train.drop(columns=drop_cols)
    X_test = test.drop(columns=[col for col in drop_cols if col in test.columns and col != 'Transported'])
    
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
        X_test[c] = X_test[c].astype('category').cat.codes
    
    print("🎯 Strategy 1: Pseudo-Labeling with High Confidence")
    
    # Train initial model to get pseudo labels
    lgbm_pseudo = LGBMClassifier(
        n_estimators=800, learning_rate=0.04, num_leaves=25,
        min_child_samples=25, random_state=42, verbose=-1
    )
    lgbm_pseudo.fit(X, y)
    test_probs = lgbm_pseudo.predict_proba(X_test)
    
    # Use very high confidence predictions as pseudo labels
    high_conf_mask = (np.max(test_probs, axis=1) > 0.95)  # 95% confidence
    print(f"   High confidence samples: {high_conf_mask.sum()}/{len(test_probs)} ({high_conf_mask.mean()*100:.1f}%)")
    
    if high_conf_mask.sum() > 50:
        pseudo_labels = np.argmax(test_probs[high_conf_mask], axis=1)
        X_pseudo = X_test[high_conf_mask]
        
        # Combine with training data
        X_combined = pd.concat([X, X_pseudo])
        y_combined = np.concatenate([y, pseudo_labels])
        
        # Retrain with pseudo labels
        lgbm_pseudo_retrain = LGBMClassifier(
            n_estimators=1000, learning_rate=0.035, num_leaves=22,
            min_child_samples=28, random_state=123, verbose=-1
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pseudo_scores = cross_val_score(lgbm_pseudo_retrain, X_combined, y_combined, cv=cv, scoring='accuracy')
        print(f"   Pseudo-labeling CV: {pseudo_scores.mean():.5f}")
        
        # Create pseudo submission
        lgbm_pseudo_retrain.fit(X_combined, y_combined)
        pseudo_preds = lgbm_pseudo_retrain.predict(X_test).astype(bool)
        
        submission_pseudo = pd.DataFrame({
            'PassengerId': X_test.index.astype(str),
            'Transported': pseudo_preds
        })
        submission_pseudo.to_csv('submission_pseudo_0813.csv', index=False)
        print("   ✅ Created submission_pseudo_0813.csv")
    
    print("\n🎯 Strategy 2: Multi-Seed Ensemble with Voting")
    
    # Train same architecture with different random seeds
    seeds = [42, 123, 456, 789, 2024, 3030, 4040]
    seed_preds = []
    
    for seed in seeds:
        lgbm_seed = LGBMClassifier(
            n_estimators=900,
            learning_rate=0.035,
            num_leaves=24,
            min_child_samples=26,
            subsample=0.82,
            colsample_bytree=0.82,
            reg_alpha=0.08,
            reg_lambda=0.08,
            random_state=seed,
            verbose=-1
        )
        lgbm_seed.fit(X, y)
        seed_preds.append(lgbm_seed.predict_proba(X_test)[:, 1])
    
    # Majority voting (more conservative)
    hard_votes = []
    for seed_prob in seed_preds:
        hard_votes.append((seed_prob > 0.5).astype(int))
    
    majority_preds = (np.mean(hard_votes, axis=0) > 0.5).astype(bool)
    
    submission_majority = pd.DataFrame({
        'PassengerId': X_test.index.astype(str),
        'Transported': majority_preds
    })
    submission_majority.to_csv('submission_majority_0813.csv', index=False)
    print("   ✅ Created submission_majority_0813.csv")
    
    # Probability averaging (less conservative)
    avg_probs = np.mean(seed_preds, axis=0)
    prob_avg_preds = (avg_probs > 0.5).astype(bool)
    
    submission_prob_avg = pd.DataFrame({
        'PassengerId': X_test.index.astype(str),
        'Transported': prob_avg_preds
    })
    submission_prob_avg.to_csv('submission_prob_avg_multi_0813.csv', index=False)
    print("   ✅ Created submission_prob_avg_multi_0813.csv")
    
    print("\n🎯 Strategy 3: Gradient Boosting Parameter Sweep")
    
    # Test multiple parameter combinations
    param_combinations = [
        {'n_estimators': 1200, 'learning_rate': 0.025, 'num_leaves': 20, 'min_child_samples': 30},
        {'n_estimators': 1000, 'learning_rate': 0.03, 'num_leaves': 22, 'min_child_samples': 28},
        {'n_estimators': 800, 'learning_rate': 0.04, 'num_leaves': 25, 'min_child_samples': 25},
        {'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 18, 'min_child_samples': 35}
    ]
    
    best_score = 0
    best_params = None
    best_model = None
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, params in enumerate(param_combinations):
        print(f"   Testing parameter set {i+1}/4...")
        
        model = LGBMClassifier(
            **params,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        
        print(f"      CV Score: {mean_score:.5f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_model = model
    
    print(f"   Best parameter set CV: {best_score:.5f}")
    print(f"   Best parameters: {best_params}")
    
    # Train best model and create submission
    best_model.fit(X, y)
    best_param_preds = best_model.predict(X_test).astype(bool)
    
    submission_best_params = pd.DataFrame({
        'PassengerId': X_test.index.astype(str),
        'Transported': best_param_preds
    })
    submission_best_params.to_csv('submission_best_params_0813.csv', index=False)
    print("   ✅ Created submission_best_params_0813.csv")
    
    print("\n🎯 Strategy 4: Feature Selection + Simple Models")
    
    # Ultra-conservative approach with fewer features
    from sklearn.feature_selection import RFE
    
    # Select top 30 features using RFE
    lgbm_selector = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    rfe = RFE(lgbm_selector, n_features_to_select=30, step=10)
    X_rfe = rfe.fit_transform(X, y)
    X_test_rfe = rfe.transform(X_test)
    
    print(f"   Selected {X_rfe.shape[1]} features with RFE")
    
    # Simple but robust model on selected features
    simple_model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=42,
        verbose=-1
    )
    
    simple_scores = cross_val_score(simple_model, X_rfe, y, cv=cv, scoring='accuracy')
    print(f"   Simple model CV: {simple_scores.mean():.5f}")
    
    simple_model.fit(X_rfe, y)
    simple_preds = simple_model.predict(X_test_rfe).astype(bool)
    
    submission_simple = pd.DataFrame({
        'PassengerId': X_test.index.astype(str),
        'Transported': simple_preds
    })
    submission_simple.to_csv('submission_simple_0813.csv', index=False)
    print("   ✅ Created submission_simple_0813.csv")
    
    print(f"\n📁 Additional submissions created:")
    print(f"   🔄 submission_pseudo_0813.csv")
    print(f"   🗳️ submission_majority_0813.csv")
    print(f"   📊 submission_prob_avg_multi_0813.csv")
    print(f"   🎛️ submission_best_params_0813.csv")
    print(f"   🎯 submission_simple_0813.csv")

if __name__ == "__main__":
    print("🚀 COMPLETE 0.813 STRATEGY")
    print("="*50)
    
    # Main approach
    model, score = ultimate_ensemble_0813()
    
    # Additional strategies if needed
    additional_advanced_strategies()
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"Main ensemble CV: {score*100:.3f}%")
    print(f"Target: 81.3%")
    print(f"Current best: 80.617%")
    
    if score > 0.813:
        print(f"🎉 SUCCESS! Main approach beats target!")
    else:
        print(f"📈 Try all {9} submissions - different approaches may work better on test set!")
    
    print(f"\n🎯 SUBMISSION PRIORITY ORDER:")
    print(f"1. submission_ensemble_0813.csv (weighted ensemble)")
    print(f"2. submission_main_stack_0813.csv (main stacking)")
    print(f"3. submission_majority_0813.csv (conservative voting)")
    print(f"4. submission_best_params_0813.csv (optimized single model)")
    print(f"5. submission_simple_0813.csv (robust simple approach)")
    print(f"\nGood luck reaching 0.813! 🍀")

