import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def kmeans_data_filling(df, n_clusters=8):
    """
    Use KMeans clustering to intelligently fill missing values
    """
    df = df.copy()
    
    # First, create basic features for clustering
    df['GroupId'] = df.index.str.split('_').str[0]
    df['GroupSize'] = df.groupby('GroupId')['HomePlanet'].transform('size')
    
    # Parse cabin info
    cabin = df['Cabin'].str.split('/', expand=True)
    df['CabinDeck'] = cabin[0]
    df['CabinNum'] = pd.to_numeric(cabin[1], errors='coerce')
    df['CabinSide'] = cabin[2]
    
    # Spending columns
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    
    # Create clustering features (using available data)
    cluster_features = []
    
    # Age (fill with median first for clustering)
    df['Age_temp'] = df['Age'].fillna(df['Age'].median())
    cluster_features.append('Age_temp')
    
    # Cabin number (fill with median first)
    df['CabinNum_temp'] = df['CabinNum'].fillna(df['CabinNum'].median())
    cluster_features.append('CabinNum_temp')
    
    # Group size
    cluster_features.append('GroupSize')
    
    # Spending (fill with 0 first)
    for col in spend_cols:
        df[f'{col}_temp'] = df[col].fillna(0)
        cluster_features.append(f'{col}_temp')
    
    # Boolean features (fill with mode)
    for col in ['CryoSleep', 'VIP']:
        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else False
        df[f'{col}_temp'] = df[col].fillna(mode_val).map({True:1, False:0, 'True':1, 'False':0})
        cluster_features.append(f'{col}_temp')
    
    # Categorical features (encode for clustering)
    for col in ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide']:
        df[f'{col}_encoded'] = df[col].astype('category').cat.codes
        cluster_features.append(f'{col}_encoded')
    
    print(f"🎯 Using {len(cluster_features)} features for KMeans clustering")
    
    # Standardize features for clustering
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_features].fillna(0))
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster)
    
    print(f"📊 Created {n_clusters} clusters for data filling")
    
    # Now use cluster-based filling for missing values
    
    # 1. Age filling by cluster
    if df['Age'].isna().any():
        age_by_cluster = df.groupby('Cluster')['Age'].median()
        missing_age = df['Age'].isna()
        for cluster in df[missing_age]['Cluster'].unique():
            cluster_age = age_by_cluster.get(cluster, df['Age'].median())
            df.loc[missing_age & (df['Cluster'] == cluster), 'Age'] = cluster_age
        print(f"✅ Filled {missing_age.sum()} missing Age values using cluster medians")
    
    # 2. Spending columns by cluster (considering CryoSleep)
    for col in spend_cols:
        if df[col].isna().any():
            missing_spend = df[col].isna()
            for cluster in df[missing_spend]['Cluster'].unique():
                cluster_data = df[df['Cluster'] == cluster]
                
                # If most people in cluster are in cryosleep, fill with 0
                cryo_rate = cluster_data['CryoSleep_temp'].mean()
                if cryo_rate > 0.7:  # 70% in cryosleep
                    fill_value = 0
                else:
                    fill_value = cluster_data[col].median()
                    if pd.isna(fill_value):
                        fill_value = 0
                
                df.loc[missing_spend & (df['Cluster'] == cluster), col] = fill_value
            
            print(f"✅ Filled {missing_spend.sum()} missing {col} values using cluster patterns")
    
    # 3. Boolean columns by cluster mode
    for col in ['CryoSleep', 'VIP']:
        if df[col].isna().any():
            missing_bool = df[col].isna()
            for cluster in df[missing_bool]['Cluster'].unique():
                cluster_data = df[df['Cluster'] == cluster]
                cluster_mode = cluster_data[col].mode()
                fill_value = cluster_mode[0] if len(cluster_mode) > 0 else False
                df.loc[missing_bool & (df['Cluster'] == cluster), col] = fill_value
            print(f"✅ Filled {missing_bool.sum()} missing {col} values using cluster modes")
    
    # 4. Categorical columns by cluster mode
    for col in ['HomePlanet', 'Destination']:
        if df[col].isna().any():
            missing_cat = df[col].isna()
            for cluster in df[missing_cat]['Cluster'].unique():
                cluster_data = df[df['Cluster'] == cluster]
                cluster_mode = cluster_data[col].mode()
                fill_value = cluster_mode[0] if len(cluster_mode) > 0 else df[col].mode()[0]
                df.loc[missing_cat & (df['Cluster'] == cluster), col] = fill_value
            print(f"✅ Filled {missing_cat.sum()} missing {col} values using cluster modes")
    
    # 5. Cabin components
    for col in ['CabinDeck', 'CabinSide']:
        if df[col].isna().any():
            missing_cabin = df[col].isna()
            for cluster in df[missing_cabin]['Cluster'].unique():
                cluster_data = df[df['Cluster'] == cluster]
                cluster_mode = cluster_data[col].mode()
                fill_value = cluster_mode[0] if len(cluster_mode) > 0 else 'Unknown'
                df.loc[missing_cabin & (df['Cluster'] == cluster), col] = fill_value
            print(f"✅ Filled {missing_cabin.sum()} missing {col} values using cluster modes")
    
    if df['CabinNum'].isna().any():
        missing_cabin_num = df['CabinNum'].isna()
        cabin_num_by_cluster = df.groupby('Cluster')['CabinNum'].median()
        for cluster in df[missing_cabin_num]['Cluster'].unique():
            fill_value = cabin_num_by_cluster.get(cluster, df['CabinNum'].median())
            df.loc[missing_cabin_num & (df['Cluster'] == cluster), 'CabinNum'] = fill_value
        print(f"✅ Filled {missing_cabin_num.sum()} missing CabinNum values using cluster medians")
    
    # Clean up temporary columns
    temp_cols = [col for col in df.columns if col.endswith('_temp') or col.endswith('_encoded')]
    df = df.drop(columns=temp_cols)
    
    return df

def stacking3_feature_engineering(df):
    """
    Your proven stacking3 feature engineering approach
    """
    df = df.copy()
    
    # Group features
    df['GroupId'] = df.index.str.split('_').str[0]
    df['GroupSize'] = df.groupby('GroupId')['HomePlanet'].transform('size')
    df['IsAlone'] = (df['GroupSize']==1).astype(int)
    
    # Cabin features (already filled by KMeans)
    df['DeckSide'] = df['CabinDeck'] + '_' + df['CabinSide']
    
    # Spending features
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    
    for c in spend_cols:
        df[f'log_{c}'] = np.log1p(df[c])
    
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['SpendPerPerson'] = df['TotalSpend'] / (df['GroupSize'] + 1)
    
    # Age binning
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,np.inf],
                         labels=['Child','Teen','Adult','Middle','Senior'],
                         include_lowest=True)
    
    # Boolean encoding
    for col in ['CryoSleep','VIP']:
        df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0}).astype(int)
    
    # Advanced interaction features
    df['CryoSpendFlag'] = (df['CryoSleep'] == 1) & (df['TotalSpend'] == 0)
    df['CryoSpendFlag'] = df['CryoSpendFlag'].astype(int)
    
    df['VIPHighSpender'] = (df['VIP'] == 1) & (df['TotalSpend'] > 1000)
    df['VIPHighSpender'] = df['VIPHighSpender'].astype(int)
    
    # Group spending patterns
    group_spend_stats = df.groupby('GroupId')['TotalSpend'].agg(['mean', 'std']).reset_index()
    group_spend_stats.columns = ['GroupId', 'GroupSpendMean', 'GroupSpendStd']
    group_spend_stats['GroupSpendStd'] = group_spend_stats['GroupSpendStd'].fillna(0)
    df = df.merge(group_spend_stats, on='GroupId', how='left')
    
    # Family patterns
    df['FamilyWithKids'] = ((df['GroupSize'] > 1) & (df['Age'] < 18)).astype(int)
    
    return df

def enhanced_stacking3():
    """
    Your stacking3 approach with KMeans data filling + micro improvements
    """
    print("🚀 KMEANS FILLING + STACKING3 APPROACH")
    print("="*50)
    
    # Load data
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    # KMeans-based data filling
    print("🎯 Step 1: KMeans-based intelligent data filling...")
    train = kmeans_data_filling(train)
    test = kmeans_data_filling(test)
    
    # Your proven feature engineering
    print("⚙️ Step 2: Stacking3 feature engineering...")
    train = stacking3_feature_engineering(train)
    test = stacking3_feature_engineering(test)
    
    # Original clustering for features (after filling)
    print("📊 Step 3: Feature clustering...")
    km_feats = train[['CabinNum','log_RoomService','log_FoodCourt',
                      'log_ShoppingMall','log_Spa','log_VRDeck','TotalSpend']].fillna(0)
    kmeans_feat = KMeans(n_clusters=6, random_state=42).fit(km_feats)
    train['FeatureCluster'] = kmeans_feat.predict(km_feats)
    test['FeatureCluster'] = kmeans_feat.predict(test[km_feats.columns].fillna(0))
    
    # Additional proven features
    train['MaxSpendItem'] = train[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].idxmax(axis=1)
    test['MaxSpendItem'] = test[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].idxmax(axis=1)
    
    # Frequency encodings
    for col in ['HomePlanet','Destination','DeckSide']:
        freq = train[col].value_counts() / len(train)
        train[f'{col}_Freq'] = train[col].map(freq).fillna(0)
        test[f'{col}_Freq'] = test[col].map(freq).fillna(0)
    
    # Prepare data
    y = train['Transported'].astype(int)
    drop_cols = ['Transported','Name','Cabin','GroupId','HomePlanet','Destination','Cluster']
    X = train.drop(columns=drop_cols)
    
    test_drop_cols = [col for col in drop_cols if col in test.columns and col != 'Transported']
    X_test = test.drop(columns=test_drop_cols)
    
    # Categorical encoding
    cat_cols = ['AgeBin','CabinDeck','CabinSide','DeckSide','MaxSpendItem','FeatureCluster']
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
        X_test[c] = X_test[c].astype('category').cat.codes
    
    print(f"📊 Final features: {X.shape[1]}")
    
    # Stacking3 models with micro-tuning for the extra push
    print("🎯 Step 4: Enhanced Stacking3 models...")
    
    # Model 1: Slightly more conservative HGB
    hgb1 = HistGradientBoostingClassifier(
        learning_rate=0.045,      # Slightly lower
        max_iter=650,             # More iterations
        max_leaf_nodes=30,        # Slightly smaller
        min_samples_leaf=20,      # More regularization
        l2_regularization=0.08,   # More regularization
        random_state=42
    )
    
    # Model 2: Different random seed + tuning
    hgb2 = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_iter=450,
        max_leaf_nodes=28,
        min_samples_leaf=18,
        l2_regularization=0.06,
        random_state=2024        # Different seed
    )
    
    # Model 3: Conservative LGBM
    lgbm = LGBMClassifier(
        n_estimators=700,         # More estimators
        learning_rate=0.04,       # Lower learning rate
        num_leaves=26,            # Smaller leaves
        min_child_samples=25,     # More regularization
        subsample=0.82,           # Slight subsampling
        colsample_bytree=0.88,    # Slight feature sampling
        reg_alpha=0.08,           # L1 regularization
        reg_lambda=0.08,          # L2 regularization
        random_state=123,         # Different seed
        verbose=-1
    )
    
    # Model 4: Add XGBoost for diversity
    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=4,              # Shallower trees
        min_child_weight=6,       # More regularization
        subsample=0.82,
        colsample_bytree=0.88,
        reg_alpha=0.08,
        reg_lambda=0.08,
        random_state=456,
        eval_metric='logloss'
    )
    
    # Model 5: CatBoost for robustness
    cat = CatBoostClassifier(
        iterations=500,
        learning_rate=0.045,
        depth=4,
        l2_leaf_reg=8,
        random_seed=789,
        verbose=False
    )
    
    # Stacking with stronger regularization
    stack = StackingClassifier(
        estimators=[
            ('hgb1', hgb1), ('hgb2', hgb2),
            ('lgbm', lgbm), ('xgb', xgb), ('cat', cat)
        ],
        final_estimator=LogisticRegression(
            C=0.6,                # More regularization
            max_iter=3000,
            random_state=42
        ),
        cv=8,                     # More robust CV
        passthrough=True,
        n_jobs=-1
    )
    
    # Cross-validation
    print("📊 Step 5: Cross-validation...")
    cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    scores = cross_val_score(stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"\n🏆 RESULTS:")
    print(f"   CV Accuracy: {scores.mean():.5f} ± {scores.std():.5f}")
    print(f"   Your best: 0.805")
    print(f"   Target: 0.808")
    print(f"   vs target: {(scores.mean() - 0.808)*100:+.2f}%")
    print(f"   Individual folds: {[f'{s:.4f}' for s in scores]}")
    
    # Train and predict
    print("🚀 Step 6: Training final model...")
    stack.fit(X, y)
    preds_test = stack.predict(X_test).astype(bool)
    
    # Create submission with proper PassengerId format
    submission = pd.DataFrame({
        'PassengerId': X_test.index.astype(str),  # Ensure string format
        'Transported': preds_test
    })
    submission.to_csv('submission_kmeans_stacking3.csv', index=False)
    
    print("✅ Created submission_kmeans_stacking3.csv")
    
    return stack, scores.mean(), submission

def additional_experiments():
    """
    Try a few other quick experiments that might help
    """
    print("\n🧪 ADDITIONAL EXPERIMENTS")
    print("="*30)
    
    # Experiment 1: Pseudo-labeling with high confidence predictions
    print("🎯 Experiment 1: Pseudo-labeling approach...")
    
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test = pd.read_csv('test.csv', index_col='PassengerId')
    
    # Quick preprocessing
    train = kmeans_data_filling(train)
    test = kmeans_data_filling(test)
    train = stacking3_feature_engineering(train)
    test = stacking3_feature_engineering(test)
    
    # Add feature cluster
    km_feats = train[['CabinNum','log_RoomService','log_FoodCourt',
                      'log_ShoppingMall','log_Spa','log_VRDeck','TotalSpend']].fillna(0)
    kmeans_feat = KMeans(n_clusters=6, random_state=42).fit(km_feats)
    train['FeatureCluster'] = kmeans_feat.predict(km_feats)
    test['FeatureCluster'] = kmeans_feat.predict(test[km_feats.columns].fillna(0))
    
    # Quick model to get pseudo labels
    y = train['Transported'].astype(int)
    drop_cols = ['Transported','Name','Cabin','GroupId','HomePlanet','Destination','Cluster']
    X = train.drop(columns=drop_cols)
    X_test = test.drop(columns=[col for col in drop_cols if col in test.columns and col != 'Transported'])
    
    # Encode categoricals
    cat_cols = ['AgeBin','CabinDeck','CabinSide','DeckSide','MaxSpendItem','FeatureCluster']
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
        X_test[c] = X_test[c].astype('category').cat.codes
    
    # Quick LGBM to get probabilities
    lgbm_quick = LGBMClassifier(n_estimators=500, random_state=42, verbose=-1)
    lgbm_quick.fit(X, y)
    test_probs = lgbm_quick.predict_proba(X_test)
    
    # Use high confidence predictions as pseudo labels
    high_conf_mask = (np.max(test_probs, axis=1) > 0.9)  # 90% confidence
    print(f"   High confidence samples: {high_conf_mask.sum()}/{len(test_probs)} ({high_conf_mask.mean()*100:.1f}%)")
    
    if high_conf_mask.sum() > 100:  # Only if we have enough confident predictions
        pseudo_labels = np.argmax(test_probs[high_conf_mask], axis=1)
        X_pseudo = X_test[high_conf_mask]
        
        # Combine with training data
        X_combined = pd.concat([X, X_pseudo])
        y_combined = np.concatenate([y, pseudo_labels])
        
        # Retrain with pseudo labels
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pseudo_scores = cross_val_score(lgbm_quick, X_combined, y_combined, cv=cv, scoring='accuracy')
        print(f"   Pseudo-labeling CV: {pseudo_scores.mean():.5f}")
        
        # Create pseudo submission with proper PassengerId format
        lgbm_quick.fit(X_combined, y_combined)
        pseudo_preds = lgbm_quick.predict(X_test).astype(bool)
        
        submission_pseudo = pd.DataFrame({
            'PassengerId': X_test.index.astype(str),  # Ensure string format
            'Transported': pseudo_preds
        })
        submission_pseudo.to_csv('submission_pseudo.csv', index=False)
        print("   ✅ Created submission_pseudo.csv")
    
    # Experiment 2: Simple ensemble of different random seeds
    print("\n🎯 Experiment 2: Multi-seed ensemble...")
    
    seeds = [42, 123, 456, 789, 2024]
    seed_preds = []
    
    for seed in seeds:
        lgbm_seed = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.045,
            num_leaves=28,
            random_state=seed,
            verbose=-1
        )
        lgbm_seed.fit(X, y)
        seed_preds.append(lgbm_seed.predict_proba(X_test)[:, 1])
    
    # Average predictions
    avg_probs = np.mean(seed_preds, axis=0)
    ensemble_preds = (avg_probs > 0.5).astype(bool)
    
    submission_ensemble = pd.DataFrame({
        'PassengerId': X_test.index.astype(str),  # Ensure string format
        'Transported': ensemble_preds
    })
    submission_ensemble.to_csv('submission_multiseed.csv', index=False)
    print("   ✅ Created submission_multiseed.csv")

if __name__ == "__main__":
    # Main approach
    model, score, submission = enhanced_stacking3()
    
    # Additional experiments
    additional_experiments()
    
    print(f"\n📊 SUMMARY:")
    print(f"Main approach CV: {score*100:.3f}%")
    print(f"Target: 80.8%")
    
    if score > 0.808:
        print(f"🎉 SUCCESS! Beat target by +{(score-0.808)*100:.2f}%")
    else:
        print(f"Close! Need +{(0.808-score)*100:.2f}% more")
    
    print(f"\n📁 Submissions created:")
    print(f"   🎯 submission_kmeans_stacking3.csv (main)")
    print(f"   🔄 submission_pseudo.csv (pseudo-labeling)")
    print(f"   🎲 submission_multiseed.csv (multi-seed ensemble)")
    print(f"\nTry submitting all three to see which performs best!")
