# predict_test_lgb_final.py

import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder

# 1) Feature‐engineering helper (same as training)
def fe(df):
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['SpendBucket'] = pd.qcut(
        df['TotalSpend'], 4, labels=False, duplicates='drop'
    )
    df['GroupID'] = df.index.to_series().str.split('_').str[0]
    df['FamilySize'] = df.groupby('GroupID')['GroupID'].transform('count')
    df['AgeBin'] = pd.cut(
        df['Age'],
        bins=[0,18,35,60,200],
        labels=['child','young','adult','senior']
    )
    return df

# 2) Load cleaned CSVs
train = pd.read_csv('train_cleaned.csv', index_col='PassengerId')
test  = pd.read_csv('test_cleaned.csv',  index_col='PassengerId')

# 3) Apply feature engineering
train = fe(train)
test  = fe(test)

# 4) Prepare training data
X_train = train.drop(columns=['Transported','GroupID'])
y_train = train['Transported'].astype(int)

# 5) Prepare test data (no Transported column)
X_test = test.drop(columns=['GroupID'])

# 6) One-hot encode categorical features
cat_cols = ['HomePlanet','Destination','Deck','Side','SpendBucket','AgeBin']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoder.fit(X_train[cat_cols])

def encode(df_):
    ohe = pd.DataFrame(
        encoder.transform(df_[cat_cols]),
        index=df_.index,
        columns=encoder.get_feature_names_out(cat_cols)
    )
    return pd.concat([df_.drop(columns=cat_cols), ohe], axis=1)

X_train_enc = encode(X_train)
X_test_enc  = encode(X_test)

# 7) Train final LightGBM on all training data
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
model.fit(X_train_enc, y_train)

# 8) Predict on test set
preds = model.predict(X_test_enc)

# 9) Build and save submission
submission = pd.DataFrame({
    'PassengerId': X_test_enc.index,
    'Transported': preds.astype(bool)
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv written!")

