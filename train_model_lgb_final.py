# train_model_lgb_final.py

import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# 1) Load & feature-engineer
df = pd.read_csv('train_cleaned.csv', index_col='PassengerId')

# Total spend
spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalSpend'] = df[spend_cols].sum(axis=1)

# Spend buckets (quartiles)
df['SpendBucket'] = pd.qcut(df['TotalSpend'], 4, labels=False, duplicates='drop')

# Family size
df['GroupID'] = df.index.to_series().str.split('_').str[0]
df['FamilySize'] = df.groupby('GroupID')['GroupID'].transform('count')

# Age bins
df['AgeBin'] = pd.cut(
    df['Age'],
    bins=[0, 18, 35, 60, 200],
    labels=['child', 'young', 'adult', 'senior']
)

# 2) Prepare X and y
X = df.drop(columns=['Transported', 'GroupID'])
y = df['Transported'].astype(int)

# 3) Optional local train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 4) One-hot encode categorical features
cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'SpendBucket', 'AgeBin']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoder.fit(X_train[cat_cols])

def encode(df_):
    ohe = pd.DataFrame(
        encoder.transform(df_[cat_cols]),
        index=df_.index,
        columns=encoder.get_feature_names_out(cat_cols)
    )
    return pd.concat([df_.drop(columns=cat_cols), ohe], axis=1)

X_train = encode(X_train)
X_val   = encode(X_val)

# 5) Instantiate LightGBM with tuned hyperparameters
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.014768998388832,
    num_leaves=42,
    min_child_samples=18,
    subsample=0.6721253135492927,
    colsample_bytree=0.7782544382924157,
    reg_alpha=1.4319289956277614e-08,
    reg_lambda=0.001463509485164253,
    random_state=42
)

# 6) Fit & evaluate
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

print(classification_report(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_prob))
