import re
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# 1. title extractor
def extract_title(name):
    if not isinstance(name, str) or name == '':
        return 'Unknown'
    m = re.search(r',\s*([^\.]+)\.', name)
    return m.group(1).strip() if m else 'Unknown'

# 2. load with proper dtypes
train = pd.read_csv('train.csv', dtype={'PassengerId': str})
test  = pd.read_csv('test.csv',  dtype={'PassengerId': str})
train['Transported'] = train['Transported'].map({True:1, False:0})

# 3. basic FE + flags
for df in (train, test):
    df['Name']    = df['Name'].fillna('').astype(str)
    df['Deck']    = df['Cabin'].fillna('Unknown').str.split('/').str[0]
    df['CabinMissing']       = df['Cabin'].isna().astype(int)
    df['DestinationMissing'] = df['Destination'].isna().astype(int)
    df['Title']   = df['Name'].apply(extract_title)
    df['CryoSleep'] = df['CryoSleep'].map({True:1, False:0})
    df['VIP']       = df['VIP'].map({True:1, False:0})
    df['Group']   = df['PassengerId'].str.split('_').str[0]

# 4. Group stats
train['GroupSize'] = train.groupby('Group')['PassengerId'].transform('count')
test ['GroupSize'] = test .groupby('Group')['PassengerId'].transform('count')
for df in (train, test):
    df['IsAlone'] = (df['GroupSize']==1).astype(int)

# 5. smarter Age imputation (median per Title)
title_med = train.groupby('Title')['Age'].median()
for df in (train, test):
    df['AgeMissing'] = df['Age'].isna().astype(int)
    df['Age'] = df.apply(
        lambda r: title_med[r['Title']] if pd.isna(r['Age']) else r['Age'],
        axis=1
    )

# 6. target‐encode big cats
cat_cols = ['HomePlanet','Destination','Deck','Title']
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for col in cat_cols:
    te = TargetEncoder(cols=[col], smoothing=0.2)
    train[col + '_te'] = np.nan
    for tr, va in kf.split(train, train['Transported']):
        te.fit(train.iloc[tr][col], train.iloc[tr]['Transported'])
        train.loc[va, col + '_te'] = te.transform(train.iloc[va][col])[col]
    te.fit(train[col], train['Transported'])
    test[col + '_te'] = te.transform(test[col])[col]

# 7. assemble features
drop = ['PassengerId','Name','Cabin','Group'] + cat_cols
features = [c for c in train.columns if c not in drop+['Transported']]
X, y = train[features], train['Transported']
X_test = test[features]

# 8. CatBoost CV + threshold sweep
oof = np.zeros(len(X))
preds = np.zeros(len(X_test))
for fold,(tr,va) in enumerate(kf.split(X,y),1):
    m = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.03,
        loss_function='Logloss', random_seed=42,
        early_stopping_rounds=50, verbose=100
    )
    m.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]))
    oof[va]   = m.predict_proba(X.iloc[va])[:,1]
    preds    += m.predict_proba(X_test)[:,1]/kf.n_splits

# sweep best threshold
ths = np.linspace(0.4,0.6,21)
scores = [(t, accuracy_score(y, oof>t)) for t in ths]
best_t, best_score = max(scores, key=lambda x: x[1])
print(f"Best CV acc={best_score:.5f} at threshold={best_t}")

# 9. final submission
sub = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': (preds > best_t)
})
sub['Transported'] = sub['Transported'].map({True:'True',False:'False'})
sub.to_csv('submission.csv', index=False)
