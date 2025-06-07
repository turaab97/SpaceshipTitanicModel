# predict_ensemble_test.py

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# 1) Feature engineering helper (same as before)
def fe(df):
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    df['TotalSpend']  = df[spend_cols].sum(axis=1)
    df['SpendBucket'] = pd.qcut(df['TotalSpend'], 4, labels=False, duplicates='drop')
    df['GroupID']     = df.index.to_series().str.split('_').str[0]
    df['FamilySize']  = df.groupby('GroupID')['GroupID'].transform('count')
    df['AgeBin']      = pd.cut(df['Age'], bins=[0,18,35,60,200],
                               labels=['child','young','adult','senior'])
    return df

# 2) Load cleaned CSVs
train = pd.read_csv('train_cleaned.csv', index_col='PassengerId')
test  = pd.read_csv('test_cleaned.csv',  index_col='PassengerId')

# 3) Apply feature engineering to both
train = fe(train)
test  = fe(test)

# 4) Prepare training data (X_train, y_train)
X_train = train.drop(columns=['Transported','GroupID'])
y_train = train['Transported'].astype(int)

# 5) Prepare test features
X_test = test.drop(columns=['GroupID'])

# 6) One-hot encode the same categorical cols
cat_cols = ['HomePlanet','Destination','Deck','Side','SpendBucket','AgeBin']
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=True)
# align test to train
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

# 7) Instantiate & train your tuned base models on all data
lgb = LGBMClassifier(
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
hgb = HistGradientBoostingClassifier(
    max_iter=1000,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)

lgb.fit(X_train, y_train)
hgb.fit(X_train, y_train)

# 8) Predict probabilities on test
p_lgb = lgb.predict_proba(X_test)[:,1]
p_hgb = hgb.predict_proba(X_test)[:,1]

# 9) Average them for our final “ensemble” score
p_final = (p_lgb + p_hgb) / 2

# 10) Binarize at 0.5 and build submission
submission = pd.DataFrame({
    'PassengerId': X_test.index,
    'Transported': (p_final > 0.5)
})
submission.to_csv('submission_ensemble.csv', index=False)
print("✅ submission_ensemble.csv ready for upload")

