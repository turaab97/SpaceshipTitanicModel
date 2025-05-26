# cleanup.py
'''-	Drop Name and raw cabin. Name is just an identifier, every passenger is unique- so it carries zero predictive power.  Raw cabin has some useless strings, which in my opinion would be useless for a model to do any prediction.  
-	HomePlaent, Desitnation, CrytoSleep, VIP  have null fields.   Change to Booleans instead for consistency (true and false) 
-	Split Cabin into deck, cabinNum and side (Feature extraction the deck letter (A, B, C… or “Unknown”) might correlate with “deck-level amenities” or even evacuation priority. Cabin side (S vs P) could have patterns too. Breaking it out gives the model actual axes to learn on instead of one indecipherable token).  
-	Median for spending and age, fill in missing flags.  Why median? It’s robust to outliers and keeps your distribution intact if you’ve got wild high‐rollers spending thousands. Why missing flags? The fact that a passenger didn’t buy anything (or didn’t report age) can itself be predictive—so we 1) fill in a reasonable default and 2) tack on a binary flag like RoomService_missing so the model knows “hey, this was originally unknown.”
'''
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
