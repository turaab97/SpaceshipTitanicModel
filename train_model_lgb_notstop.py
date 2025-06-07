# train_model_lgb_nostop.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load cleaned data
df = pd.read_csv('train_cleaned.csv', index_col='PassengerId')

# 2. Feature engineering
spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalSpend'] = df[spend_cols].sum(axis=1)

# Use qcut with duplicates dropped
df['SpendBucket'] = pd.qcut(
    df['TotalSpend'],
    4,
    labels=False,
    duplicates='drop'
)

# Family size
df['GroupID'] = df.index.to_series().str.split('_').str[0]
df['FamilySize'] = df.groupby('GroupID')['GroupID'].transform('count')

# Age bins
df['AgeBin'] = pd.cut(
    df['Age'],
    bins=[0, 18, 35, 60, 200],
    labels=['child', 'young', 'adult', 'senior']
)

# 3. Prepare features X and target y
X = df.drop(columns=['Transported', 'GroupID'])
y = df['Transported'].astype(int)

# 4. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 5. One-hot encode categorical columns
cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'SpendBucket', 'AgeBin']
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_val = pd.get_dummies(X_val, columns=cat_cols, drop_first=True)
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

# 6. Train LightGBM (no early stopping)
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 7. Evaluate on validation set
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

print("=== Validation Metrics ===")
print(classification_report(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_prob))

