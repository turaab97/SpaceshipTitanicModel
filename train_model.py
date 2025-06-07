import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Load cleaned data
df = pd.read_csv('train_cleaned.csv', index_col='PassengerId')

# Split into features and target
X = df.drop(columns=['Transported'])
y = df['Transported'].astype(int)

# Train/validation split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# One-hot encode categorical features
cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_val   = pd.get_dummies(X_val,   columns=cat_cols, drop_first=True)
X_val   = X_val.reindex(columns=X_train.columns, fill_value=0)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_val)
probs = model.predict_proba(X_val)[:, 1]
print(classification_report(y_val, preds))
print("ROC AUC:", roc_auc_score(y_val, probs))

