#0.808

'''The goal is to predict whether a passenger was "Transported" (a binary yes/no outcome) based on information about them.
Step-by-step:
1. Load the Data
Reads in two CSV files: train.csv (with answers) and test.csv (without answers).
Each row is a passenger, and each column is a feature (like age, spending, etc.).

2. Feature Engineering
Creates new features from the existing data to help the model learn better.
Examples:
GroupId/GroupSize/IsAlone: Figures out if a passenger is traveling alone or with a group.
Cabin features: Splits the cabin info into deck, number, and side.
Spending features: Fills missing spending with 0, creates log versions, sums up total spend, and calculates spend per person.
Age bins: Groups ages into categories like Child, Teen, Adult, etc.
Boolean features: Converts "CryoSleep" and "VIP" to 1/0.
Composite features: Combines deck and side into one feature.

3. Advanced Features
Clustering: Groups passengers into 6 clusters based on their spending and cabin number.
MaxSpendItem: Finds which spending category each passenger spent the most on.
Frequency features: Calculates how common each value is for some columns.

4. Prepare Data for Modeling
Sets up the input features (X) and the target variable (y).
Drops columns that shouldn’t be used for prediction (like names).
Converts categorical features to numbers (since models need numbers).

5. Model Building: Stacking Ensemble
Uses a StackingClassifier, which is an ensemble (combination) of several models:
HistGradientBoostingClassifier (HGB): A fast, tree-based model from scikit-learn (used twice with different settings).
LGBMClassifier: Another fast, tree-based model from LightGBM.
The outputs of these models are combined using a Logistic Regression as the final decision-maker.
This approach is called stacking and often gives better results than a single model.

6. Cross-Validation
Tests the model using cross-validation (splitting the data into 5 parts, training on 4, testing on 1, and repeating).
Prints the average accuracy and how much it varies.

7. Train on All Data & Make Submission
Trains the stacking model on all the training data.
Predicts the results for the test data.
Saves the predictions in a CSV file for submission to Kaggle.


What model does it use?
Stacking Ensemble: Combines three models (two HGB, one LGBM) and uses Logistic Regression to make the final prediction.
HGB (HistGradientBoostingClassifier): A tree-based model that works well with tabular data.
LGBM (LightGBMClassifier): Another tree-based model, very fast and accurate.
Logistic Regression: A simple model that combines the outputs of the above models.

What does the accuracy mean?
Accuracy = 0.80804 (or about 80.8%)
This means that, on average, the model correctly predicts whether a passenger was transported about 81 times out of 100.

This is measured using cross-validation, which is a reliable way to estimate how well the model will perform on new, unseen data.
Summary
The code loads and processes data, creates new features, and uses a powerful combination of machine learning models to predict an outcome.

The stacking ensemble approach helps improve accuracy.
The model achieves about 81% accuracy in predicting the outcome.

how does stacking work ? 

Stacking (short for "stacked generalization") is an ensemble learning technique in machine learning. It combines the predictions of multiple different models to make a final prediction, often resulting in better performance than any single model alone.
How does stacking work?
1. Base Models (Level 0)
You train several different models (called base models or level 0 models) on your training data.
These models can be of different types (e.g., decision trees, gradient boosting, logistic regression, etc.).
Each base model makes its own predictions.
2. Meta-Model (Level 1)
You then train a new model (called the meta-model or level 1 model) using the predictions from the base models as input features.
The meta-model learns how to best combine the base models’ predictions to make the final prediction.
3. Making Predictions
For new, unseen data:
Each base model makes a prediction.
The meta-model takes these predictions and makes the final decision.
Why does stacking work?
Different models have different strengths and weaknesses.
By combining them, stacking can "average out" their errors and capture patterns that a single model might miss.
The meta-model learns which base models to trust more in different situations.
Visual Example
Suppose you have three base models:
Model A: Decision Tree
Model B: Gradient Boosting
Model C: Logistic Regression
For a given passenger, their predictions might be:
Model A: 0.7 (probability of being transported)
Model B: 0.4
Model C: 0.9
The meta-model (say, another logistic regression) takes these three numbers as input and learns to combine them (maybe giving more weight to Model C if it’s usually more accurate).
In your code:
Base models: Two HistGradientBoostingClassifier and one LGBMClassifier.
Meta-model: LogisticRegression.
The stacking classifier automatically handles the process of training base models, collecting their predictions, and training the meta-model.
'''

# 0.808 Kaggle score example

# 1. Import necessary libraries
import pandas as pd                      # Data manipulation and CSV I/O
import numpy as np                       # Numerical operations
from sklearn.cluster import KMeans       # Clustering algorithm
from sklearn.model_selection import StratifiedKFold, cross_val_score
#   StratifiedKFold: creates balanced train/test splits
#   cross_val_score: evaluates model performance across folds
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
#   HistGradientBoostingClassifier: fast gradient-boosting model
#   StackingClassifier: combines several models into an ensemble
from lightgbm import LGBMClassifier      # LightGBM gradient-boosting model
from sklearn.linear_model import LogisticRegression
#   LogisticRegression: linear model for final stacking layer

### Kaggle score: 0.80804

# 2. Load the datasets
train = pd.read_csv('train.csv', index_col='PassengerId')  # Read training data, use PassengerId as index
test  = pd.read_csv('test.csv',  index_col='PassengerId')  # Read test data

# 3. Define feature engineering function
def engineer(df):
    df = df.copy()  # Work on a copy to avoid modifying original

    # 3.1 Group-level features
    df['GroupId']   = df.index.str.split('_').str[0]
    #   Extract group ID from PassengerId (before the underscore)
    df['GroupSize'] = df.groupby('GroupId')['HomePlanet'].transform('size')
    #   Count how many passengers share the same GroupId
    df['IsAlone']   = (df['GroupSize'] == 1).astype(int)
    #   Flag passengers traveling alone (1) vs in a group (0)

    # 3.2 Parse cabin information
    cabin = df['Cabin'].str.split('/', expand=True)
    #   Split "Deck/Number/Side" into three parts
    df['CabinDeck'] = cabin[0].fillna('Unknown')
    #   Deck letter or "Unknown"
    cn = pd.to_numeric(cabin[1], errors='coerce')
    #   Convert cabin number to numeric (invalid → NaN)
    df['CabinNum']  = cn.fillna(cn.median())
    #   Fill missing cabin numbers with median
    df['CabinSide'] = cabin[2].fillna('Unknown')
    #   Side of cabin ("P"/"S") or "Unknown"

    # 3.3 Spending features
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    df[spend_cols] = df[spend_cols].fillna(0)
    #   Replace any missing spending with 0

    # 3.4 Log-transform spend to reduce skew
    for c in spend_cols:
        df[f'log_{c}'] = np.log1p(df[c])
        #   log1p = log(1 + x), safe for zero values

    # 3.5 Total and per-person spend
    df['TotalSpend']      = df[spend_cols].sum(axis=1)
    #   Sum of all spend columns
    df['SpendPerPerson']  = df['TotalSpend'] / (df['GroupSize'] + 1)
    #   Divide by group size +1 to avoid division by zero

    # 3.6 Age handling
    df['Age'] = df['Age'].fillna(df['Age'].median())
    #   Fill missing ages with median age
    df['AgeBin'] = pd.cut(
        df['Age'],
        bins=[0,12,18,35,60,np.inf],
        labels=['Child','Teen','Adult','Middle','Senior'],
        include_lowest=True
    )
    #   Discretize age into categories

    # 3.7 Composite cabin feature
    df['DeckSide'] = df['CabinDeck'] + '_' + df['CabinSide']
    #   Combine deck and side into one feature

    # 3.8 Boolean mapping for CryoSleep and VIP
    for col in ['CryoSleep','VIP']:
        df[col] = (
            df[col]
            .map({True:1, False:0, 'True':1, 'False':0})
            .fillna(0)
            .astype(int)
        )
        #   Convert True/False (and strings) to 1/0, fill NaN as 0

    return df  # Return engineered DataFrame

# 4. Apply feature engineering
train = engineer(train)  # Transform training set
test  = engineer(test)   # Transform test set

# 5. Advanced features

# 5.1 Cluster passengers by spending pattern
km_feats = train[[
    'CabinNum','log_RoomService','log_FoodCourt',
    'log_ShoppingMall','log_Spa','log_VRDeck','TotalSpend'
]].fillna(0)
#   Features to feed into KMeans, fill NaN with 0
kmeans = KMeans(n_clusters=6, random_state=42).fit(km_feats)
#   Fit KMeans with 6 clusters
train['Cluster'] = kmeans.predict(km_feats)
#   Assign each train passenger a cluster label
test['Cluster']  = kmeans.predict(test[km_feats.columns].fillna(0))
#   Assign cluster labels to test set

# 5.2 Identify the single highest spend category for each passenger
train['MaxSpendItem'] = train[spend_cols].idxmax(axis=1)
#   Column name of the max spending service
test['MaxSpendItem']  = test[spend_cols].idxmax(axis=1)

# 5.3 Frequency encoding for some categorical features
for col in ['HomePlanet','Destination','DeckSide']:
    freq = train[col].value_counts() / len(train)
    #   Calculate how common each category is
    train[f'{col}_Freq'] = train[col].map(freq).fillna(0)
    #   Map frequency to train set
    test[f'{col}_Freq']  = test[col].map(freq).fillna(0)
    #   Map frequency to test set

# 6. Prepare feature matrices

y = train['Transported'].astype(int)
#   Target variable (0/1 transported)
drop_cols = ['Transported','Name','Cabin','GroupId','HomePlanet','Destination']
X = train.drop(columns=drop_cols)
#   Drop unused or high-cardinality raw columns
X_test = test.drop(columns=drop_cols[:-1])
#   Drop same columns (test has no Transported)

# 7. Encode remaining categorical features as integer codes
cat_cols = ['AgeBin','CabinDeck','CabinSide','DeckSide','MaxSpendItem','Cluster']
for c in cat_cols:
    X[c]      = X[c].astype('category').cat.codes
    X_test[c] = X_test[c].astype('category').cat.codes
    #   Convert each category to a numeric code

# 8. Define base models for stacking

hgb1 = HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=500, max_leaf_nodes=31, random_state=42
)
#   First HistGradientBoosting model

hgb2 = HistGradientBoostingClassifier(
    learning_rate=0.1, max_iter=300, max_leaf_nodes=31, random_state=24
)
#   Second HistGradientBoosting model with different hyperparams

lgbm = LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=0
)
#   LightGBM model

# 9. Build stacking ensemble
stack = StackingClassifier(
    estimators=[('hgb1', hgb1), ('hgb2', hgb2), ('lgbm', lgbm)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,            # 5-fold CV inside stacking
    passthrough=True,# Include original features for the final estimator
    n_jobs=-1        # Use all CPU cores
)
#   Final meta-model is Logistic Regression

# 10. Evaluate with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#   Ensure each fold has balanced Transported classes
scores = cross_val_score(
    stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1
)
#   Compute accuracy in each fold
print(f"Stacking CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
#   Print average accuracy ± standard deviation

# 11. Train on full data and predict on test set
stack.fit(X, y)
#   Fit stacking ensemble on all training data
preds_test = stack.predict(X_test).astype(bool)
#   Generate final True/False predictions

# 12. Save submission file
submission = pd.DataFrame({
    'PassengerId': X_test.index,
    'Transported': preds_test
})
submission.to_csv('submission_stack3.csv', index=False)
#   Write predictions for Kaggle submission
print("✔ Done! submission_stack3.csv generated.")
#   Notify that the file is ready
