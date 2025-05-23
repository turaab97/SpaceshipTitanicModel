# cleanup.py

import os
import pandas as pd

# Paths (assumes you run this from the folder containing train.csv & test.csv)
TRAIN_CSV  = os.path.join(os.getcwd(), 'train.csv')
TEST_CSV   = os.path.join(os.getcwd(), 'test.csv')
OUT_TRAIN  = os.path.join(os.getcwd(), 'train_cleaned.csv')
OUT_TEST   = os.path.join(os.getcwd(), 'test_cleaned.csv')

# --- Clean train.csv ---
df = pd.read_csv(TRAIN_CSV)
df = df.drop(columns=['Name']).set_index('PassengerId')
df['HomePlanet']  = df['HomePlanet'].fillna('Unknown')
df['Destination'] = df['Destination'].fillna('Unknown')
df['CryoSleep']   = df['CryoSleep'].fillna(False).astype(bool)
df['VIP']         = df['VIP'].fillna(False).astype(bool)
df['Cabin']       = df['Cabin'].fillna('Unknown/0/Unknown')
df[['Deck','CabinNum','Side']] = df['Cabin'].str.split('/', expand=True)
df = df.drop(columns=['Cabin'])
num_cols = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
for col in num_cols:
    df[f'{col}_missing'] = df[col].isnull().astype(int)
    df[col] = df[col].fillna(df[col].median())
df.to_csv(OUT_TRAIN)
print(f"Saved cleaned train → {OUT_TRAIN}")

# --- Clean test.csv ---
t = pd.read_csv(TEST_CSV)
t = t.drop(columns=['Name']).set_index('PassengerId')
t['HomePlanet']  = t['HomePlanet'].fillna('Unknown')
t['Destination'] = t['Destination'].fillna('Unknown')
t['CryoSleep']   = t['CryoSleep'].fillna(False).astype(bool)
t['VIP']         = t['VIP'].fillna(False).astype(bool)
t['Cabin']       = t['Cabin'].fillna('Unknown/0/Unknown')
t[['Deck','CabinNum','Side']] = t['Cabin'].str.split('/', expand=True)
t = t.drop(columns=['Cabin'])
for col in num_cols:
    t[f'{col}_missing'] = t[col].isnull().astype(int)
    t[col] = t[col].fillna(df[col].median())  # use train medians
t.to_csv(OUT_TEST)
print(f"Saved cleaned test  → {OUT_TEST}")
