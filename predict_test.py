import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Load cleaned train & test
train = pd.read_csv('train_cleaned.csv', index_col='PassengerId')
test  = pd.read_csv('test_cleaned.csv',  index_col='PassengerId')

# 2. Split features / target
X_train = train.drop(columns=['Transported'])
y_train = train['Transported'].astype(int)

# 3. Fit final model on all training data
model = LogisticRegression(max_iter=5000, solver='saga')
# One-hot encode same way as before
cat_cols = ['HomePlanet','Destination','Deck','Side']
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)

model.fit(X_train, y_train)

# 4. Prepare test set
X_test = pd.get_dummies(test, columns=cat_cols, drop_first=True)
# Align to train columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 5. Predict and save
preds = model.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': X_test.index,
    'Transported': preds.astype(bool)
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv ready")

