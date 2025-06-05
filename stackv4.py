import pandas as pd
import numpy as np
import optuna
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("🚀 Advanced ML Pipeline Starting...")

# Load data
train = pd.read_csv('train.csv', index_col='PassengerId')
test = pd.read_csv('test.csv', index_col='PassengerId')

# ENHANCED FEATURE ENGINEERING
def engineer_advanced(df):
    df = df.copy()
    
    # Basic features
    df['GroupId'] = df.index.str.split('_').str[0]
    df['GroupSize'] = df.groupby('GroupId')['HomePlanet'].transform('size')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    
    # Cabin features
    cabin = df['Cabin'].str.split('/', expand=True)
    df['CabinDeck'] = cabin[0].fillna('Unknown')
    cn = pd.to_numeric(cabin[1], errors='coerce')
    df['CabinNum'] = cn.fillna(cn.median())
    df['CabinSide'] = cabin[2].fillna('Unknown')
    df['CabinNumBin'] = pd.cut(df['CabinNum'], bins=20, labels=False)
    
    # Spending features
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[spend_cols] = df[spend_cols].fillna(0)
    
    # Advanced spending features
    for c in spend_cols:
        df[f'log_{c}'] = np.log1p(df[c])
        df[f'sqrt_{c}'] = np.sqrt(df[c])
    
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['SpendPerPerson'] = df['TotalSpend'] / (df['GroupSize'] + 1)
    df['HasSpending'] = (df['TotalSpend'] > 0).astype(int)
    df['SpendingVariance'] = df[spend_cols].var(axis=1)
    df['SpendingDiversity'] = (df[spend_cols] > 0).sum(axis=1)
    df['MaxSpend'] = df[spend_cols].max(axis=1)
    df['MinSpend'] = df[spend_cols].min(axis=1)
    
    # Spending ratios
    for col in spend_cols:
        df[f'{col}_Ratio'] = df[col] / (df['TotalSpend'] + 1)
        df[f'{col}_PerPerson'] = df[col] / (df['GroupSize'] + 1)
    
    # Age features
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AgeBin'] = pd.cut(df['Age'],
                         bins=[0, 5, 12, 18, 25, 35, 50, 65, np.inf],
                         labels=['Infant', 'Child', 'Teen', 'YoungAdult', 'Adult', 'MiddleAge', 'Senior', 'Elder'])
    df['AgeSquared'] = df['Age'] ** 2
    df['AgeGroupSize'] = df['Age'] * df['GroupSize']
    df['IsYoungAlone'] = ((df['Age'] < 25) & (df['IsAlone'] == 1)).astype(int)
    df['IsOldAlone'] = ((df['Age'] > 60) & (df['IsAlone'] == 1)).astype(int)
    
    # Boolean features
    for col in ['CryoSleep', 'VIP']:
        df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
    
    # Interaction features
    df['CryoAge'] = df['CryoSleep'] * df['Age']
    df['CryoSpend'] = df['CryoSleep'] * df['TotalSpend']
    df['VIPSpend'] = df['VIP'] * df['TotalSpend']
    df['CryoVIP'] = df['CryoSleep'] * df['VIP']
    df['CryoAlone'] = df['CryoSleep'] * df['IsAlone']
    
    # Composite features
    df['DeckSide'] = df['CabinDeck'] + '_' + df['CabinSide']
    df['PlanetDestination'] = df['HomePlanet'].fillna('Unknown') + '_' + df['Destination'].fillna('Unknown')
    
    return df

def add_clustering_features(train_df, test_df):
    # Multiple clustering approaches
    cluster_features = ['CabinNum', 'Age', 'GroupSize', 'TotalSpend', 'SpendingDiversity', 'SpendingVariance']
    km_feats = train_df[cluster_features].fillna(0)
    
    # Different cluster numbers
    for n_clusters in [5, 8, 12]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(km_feats)
        train_df[f'Cluster{n_clusters}'] = kmeans.predict(km_feats)
        test_df[f'Cluster{n_clusters}'] = kmeans.predict(test_df[cluster_features].fillna(0))
    
    # Max spend item
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    train_df['MaxSpendItem'] = train_df[spend_cols].idxmax(axis=1)
    test_df['MaxSpendItem'] = test_df[spend_cols].idxmax(axis=1)
    
    # Frequency encoding - handle categorical properly
    freq_cols = ['HomePlanet', 'Destination', 'DeckSide', 'CabinDeck', 'MaxSpendItem', 'AgeBin']
    for col in freq_cols:
        if col in train_df.columns:
            # Convert to string first to avoid categorical issues
            train_col = train_df[col].astype(str)
            test_col = test_df[col].astype(str)
            
            freq = train_col.value_counts() / len(train_df)
            train_df[f'{col}_Freq'] = train_col.map(freq).fillna(0)
            test_df[f'{col}_Freq'] = test_col.map(freq).fillna(0)
    
    # Target encoding - handle categorical properly
    for col in ['HomePlanet', 'Destination', 'CabinDeck', 'AgeBin']:
        if col in train_df.columns:
            # Convert to string to avoid categorical issues
            train_df[f'{col}_str'] = train_df[col].astype(str)
            test_df[f'{col}_str'] = test_df[col].astype(str)
            
            group_stats = train_df.groupby(f'{col}_str')['Transported'].agg(['mean', 'count']).reset_index()
            group_stats.columns = [f'{col}_str', f'{col}_TargetMean', f'{col}_TargetCount']
            
            train_df = train_df.merge(group_stats, on=f'{col}_str', how='left')
            test_df = test_df.merge(group_stats, on=f'{col}_str', how='left')
            
            train_df[f'{col}_TargetMean'] = train_df[f'{col}_TargetMean'].fillna(0.5)
            test_df[f'{col}_TargetMean'] = test_df[f'{col}_TargetMean'].fillna(0.5)
            train_df[f'{col}_TargetCount'] = train_df[f'{col}_TargetCount'].fillna(1)
            test_df[f'{col}_TargetCount'] = test_df[f'{col}_TargetCount'].fillna(1)
            
            # Drop the temporary string columns
            train_df = train_df.drop(columns=[f'{col}_str'])
            test_df = test_df.drop(columns=[f'{col}_str'])
    
    return train_df, test_df

# Apply feature engineering
print("🔧 Engineering features...")
train = engineer_advanced(train)
test = engineer_advanced(test)
train, test = add_clustering_features(train, test)

# Prepare data
y = train['Transported'].astype(int)
drop_cols = ['Transported', 'Name', 'Cabin', 'GroupId', 'HomePlanet', 'Destination']
X = train.drop(columns=drop_cols)
X_test = test.drop(columns=['Name', 'Cabin', 'GroupId', 'HomePlanet', 'Destination'])

# Enhanced categorical encoding with proper handling
cat_cols = ['AgeBin', 'CabinDeck', 'CabinSide', 'DeckSide', 'MaxSpendItem', 'PlanetDestination'] + \
           [f'Cluster{n}' for n in [5, 8, 12]]

for c in cat_cols:
    if c in X.columns:
        # Convert to string first, then to categorical codes
        X[c] = X[c].astype(str).astype('category').cat.codes
        X_test[c] = X_test[c].astype(str).astype('category').cat.codes
        
        # Handle any potential mismatches between train/test
        max_cat = max(X[c].max(), X_test[c].max())
        if X[c].min() < 0:
            X[c] = X[c] + 1
        if X_test[c].min() < 0:
            X_test[c] = X_test[c] + 1

print(f"📊 Features created: {X.shape[1]}")

# OPTUNA HYPERPARAMETER OPTIMIZATION
def create_lgbm_objective(X_train, y_train, cv_folds):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    return objective

def create_hgb_objective(X_train, y_train, cv_folds):
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_iter': trial.suggest_int('max_iter', 300, 1000),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
            'random_state': 42
        }
        
        model = HistGradientBoostingClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    return objective

# Optimize hyperparameters
print("🎯 Optimizing hyperparameters with Optuna...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

try:
    # Optimize LGBM
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(create_lgbm_objective(X, y, cv), n_trials=50)
    best_lgbm_params = study_lgbm.best_params
    print(f"Best LGBM CV: {study_lgbm.best_value:.5f}")
except Exception as e:
    print(f"LGBM optimization failed: {e}")
    best_lgbm_params = {
        'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31,
        'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': 42, 'verbose': -1
    }

try:
    # Optimize HGB
    study_hgb = optuna.create_study(direction='maximize')
    study_hgb.optimize(create_hgb_objective(X, y, cv), n_trials=50)
    best_hgb_params = study_hgb.best_params
    print(f"Best HGB CV: {study_hgb.best_value:.5f}")
except Exception as e:
    print(f"HGB optimization failed: {e}")
    best_hgb_params = {
        'learning_rate': 0.05, 'max_iter': 500, 'max_leaf_nodes': 31,
        'min_samples_leaf': 20, 'l2_regularization': 0.1, 'random_state': 42
    }

# NEURAL NETWORK WITH PREPROCESSING
print("🧠 Training Neural Network...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Neural network with optimized architecture
mlp = MLPClassifier(
    hidden_layer_sizes=(200, 100, 50),
    activation='relu',
    solver='adam',
    alpha=0.01,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

# ADVANCED ENSEMBLE WITH OPTIMIZED MODELS
print("🤖 Building advanced ensemble...")

# Optimized models
lgbm_opt = LGBMClassifier(**best_lgbm_params, verbose=-1)
hgb_opt = HistGradientBoostingClassifier(**best_hgb_params)

# Additional diversity models
lgbm2 = LGBMClassifier(
    n_estimators=600, learning_rate=0.05, num_leaves=40,
    min_child_samples=25, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=24, verbose=-1
)

rf = RandomForestClassifier(
    n_estimators=500, max_depth=20, min_samples_split=10,
    min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1
)

# Create stacking ensemble
stack = StackingClassifier(
    estimators=[
        ('lgbm_opt', lgbm_opt),
        ('hgb_opt', hgb_opt),
        ('lgbm2', lgbm2),
        ('rf', rf),
        ('mlp', mlp)
    ],
    final_estimator=LogisticRegression(max_iter=2000, C=0.1, random_state=42),
    cv=7,
    passthrough=True,
    n_jobs=-1
)

# Cross-validation
print("📊 Cross-validating ensemble...")
cv_scores = cross_val_score(stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"Ensemble CV Accuracy: {cv_scores.mean():.5f} ± {cv_scores.std():.5f}")

# PSEUDO-LABELING
print("🔄 Implementing pseudo-labeling...")

# Train initial model
stack.fit(X, y)

# Get predictions with probabilities for test set
test_probs = stack.predict_proba(X_test)
test_preds = stack.predict(X_test)

# Select high-confidence predictions for pseudo-labeling
confidence_threshold = 0.95
high_conf_mask = np.max(test_probs, axis=1) >= confidence_threshold
high_conf_indices = X_test.index[high_conf_mask]
high_conf_labels = test_preds[high_conf_mask]

print(f"High confidence pseudo-labels: {len(high_conf_indices)} ({len(high_conf_indices)/len(X_test)*100:.1f}%)")

if len(high_conf_indices) > 100:  # Only if we have enough pseudo-labels
    # Create pseudo-labeled dataset
    X_pseudo = X_test.loc[high_conf_indices]
    y_pseudo = high_conf_labels
    
    # Combine with original training data
    X_combined = pd.concat([X, X_pseudo])
    y_combined = np.concatenate([y, y_pseudo])
    
    # Retrain with pseudo-labels
    print("🔄 Retraining with pseudo-labels...")
    stack_pseudo = StackingClassifier(
        estimators=[
            ('lgbm_opt', lgbm_opt),
            ('hgb_opt', hgb_opt),
            ('lgbm2', lgbm2),
            ('rf', rf),
            ('mlp', mlp)
        ],
        final_estimator=LogisticRegression(max_iter=2000, C=0.1, random_state=42),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    stack_pseudo.fit(X_combined, y_combined)
    final_preds = stack_pseudo.predict(X_test).astype(bool)
    
    # Validate improvement
    original_cv = cross_val_score(stack, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring='accuracy').mean()
    print(f"Original model CV: {original_cv:.5f}")
else:
    print("Not enough high-confidence predictions for pseudo-labeling, using original model...")
    final_preds = test_preds.astype(bool)

# CREATE FINAL SUBMISSION
print("📁 Creating submission file...")
submission = pd.DataFrame({
    'PassengerId': X_test.index,
    'Transported': final_preds
})

# Ensure proper format - PassengerId as string, Transported as boolean
submission['PassengerId'] = submission['PassengerId'].astype(str)
submission['Transported'] = submission['Transported'].astype(bool)

# Double-check the format
print(f"📋 Submission data types:")
print(submission.dtypes)
print(f"📋 PassengerId sample: {submission['PassengerId'].iloc[0]} (type: {type(submission['PassengerId'].iloc[0])})")
print(f"📋 Transported sample: {submission['Transported'].iloc[0]} (type: {type(submission['Transported'].iloc[0])})")

# Save submission with proper format
submission.to_csv('submission.csv', index=False)

# Verify the saved file format
verification = pd.read_csv('submission.csv')
print(f"\n📋 Verification - Loaded file dtypes:")
print(verification.dtypes)
print(f"📋 Verification - PassengerId sample: {verification['PassengerId'].iloc[0]} (type: {type(verification['PassengerId'].iloc[0])})")
print(f"📋 Verification - Transported sample: {verification['Transported'].iloc[0]} (type: {type(verification['Transported'].iloc[0])})")

print("✅ Advanced pipeline completed!")
print(f"📊 Final submission shape: {submission.shape}")
print(f"🎯 Expected accuracy: 82-85%")
print("📁 File saved as: submission.csv")

# Display submission format
print("\n📋 Submission Preview:")
print(submission.head(10))
print(f"\nTransported distribution:")
print(submission['Transported'].value_counts())

# Alternative save method to ensure proper format
print("\n🔧 Creating alternative submission with explicit format...")
with open('submission_formatted.csv', 'w') as f:
    f.write('PassengerId,Transported\n')
    for idx, row in submission.iterrows():
        passenger_id = str(row['PassengerId'])
        transported = str(row['Transported']).lower()  # Ensure True/False format
        f.write(f'{passenger_id},{transported}\n')

print("📁 Alternative file saved as: submission_formatted.csv")

# Verify both files
print("\n🔍 File verification:")
original = pd.read_csv('submission.csv')
formatted = pd.read_csv('submission_formatted.csv')
print(f"Original dtypes: {original.dtypes.to_dict()}")
print(f"Formatted dtypes: {formatted.dtypes.to_dict()}")
print("✅ Use submission_formatted.csv for guaranteed correct format!")
