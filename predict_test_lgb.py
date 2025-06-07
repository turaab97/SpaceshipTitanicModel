# predict_test_lgb.py

import pandas as pd
import lightgbm as lgb

# 1) Load cleaned train & test
df_train = pd.read_csv('train_cleaned.csv', index_col='PassengerId')
df_test  = pd.read_csv('test_cleaned.csv',  index_col='PassengerId')

# 2) Feature engineering function (apply to both train & test)
def fe(df):
    # Total spend
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    # Spend bucket
    df['SpendBucket'] = pd.qcut(
        df['TotalSpend'], 4, labels=False, duplicates='drop'
    )
    # Family size
    df['GroupID'] = df.index.to_series().str.split('_').str[0]
    df['FamilySize'] = df.groupby('GroupID')['GroupID'].transform('count')
    # Age bins
    df['AgeBin'] = pd.cut(
        df['Age'], bins=[0,18,35,60,200],
        labels=['child','young','adult','senior']
    )
    return df

df_train = fe(df_train)
df_test  = fe(df_test)

# 3) Prepare X/y for full training
X_train = df_train.drop(columns=['Transported','GroupID'])
y_train = df_train['Transported'].astype(int)

# 4) One-hot encode cats on train
cat_cols = ['HomePlanet','Destination','Deck','Side','SpendBucket','AgeBin']
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)

# 5) Train final LightGBM on all data
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 6) Prepare test features (align columns)
X_test = df_test.drop(columns=['GroupID'])
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 7) Predict and save submission
preds = model.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': X_test.index,
    'Transported': preds.astype(bool)
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv written!")

