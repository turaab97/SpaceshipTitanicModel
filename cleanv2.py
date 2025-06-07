import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def advanced_data_cleaning(train_path='train.csv', test_path='test.csv'):
    """
    Fixed advanced data cleaning pipeline
    """
    print("🔍 Loading raw data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"📊 Raw data: Train {train.shape}, Test {test.shape}")
    
    # Create combined dataset for consistent processing
    train['is_train'] = True
    test['is_train'] = False
    if 'Transported' not in test.columns:
        test['Transported'] = np.nan
    
    # Ensure all data is string type initially to avoid categorical issues
    for col in ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Cabin']:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
    
    combined = pd.concat([train, test], ignore_index=True)
    print(f"📊 Combined shape: {combined.shape}")
    
    # STEP 1: Extract group information
    print("\n🏠 Extracting group information...")
    combined['GroupId'] = combined['PassengerId'].str.split('_').str[0]
    combined['PersonInGroup'] = combined['PassengerId'].str.split('_').str[1].astype(int)
    
    # Calculate group statistics
    group_stats = combined.groupby('GroupId').agg({
        'PassengerId': 'count',
        'Age': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    group_stats.columns = ['GroupSize', 'GroupAge_mean', 'GroupAge_std', 'GroupAge_min', 'GroupAge_max']
    group_stats = group_stats.reset_index()
    group_stats['GroupAge_std'] = group_stats['GroupAge_std'].fillna(0)
    
    # Merge back to main dataset
    combined = combined.merge(group_stats, on='GroupId', how='left')
    combined['IsAlone'] = (combined['GroupSize'] == 1).astype(int)
    
    print(f"✅ Group analysis complete. Found {combined['GroupId'].nunique()} groups")
    
    # STEP 2: Handle spending columns first
    print("\n💰 Processing spending data...")
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Fill spending NaN with 0
    for col in spending_cols:
        combined[col] = combined[col].fillna(0)
    
    combined['TotalSpend'] = combined[spending_cols].sum(axis=1)
    combined['HasAnySpending'] = (combined['TotalSpend'] > 0).astype(int)
    
    # STEP 3: CryoSleep imputation using spending patterns
    print("\n❄️ CryoSleep imputation...")
    
    def impute_cryosleep_simple(row):
        cryo_val = str(row['CryoSleep']).lower()
        if cryo_val in ['true', '1', 'yes']:
            return 1
        elif cryo_val in ['false', '0', 'no']:
            return 0
        elif row['TotalSpend'] == 0:
            return 1  # Zero spending likely means CryoSleep
        elif row['TotalSpend'] > 0:
            return 0  # Any spending means not CryoSleep
        else:
            return 0  # Default to not CryoSleep
    
    combined['CryoSleep'] = combined.apply(impute_cryosleep_simple, axis=1)
    
    # Force CryoSleep passengers to have zero spending
    cryo_mask = combined['CryoSleep'] == 1
    for col in spending_cols:
        combined.loc[cryo_mask, col] = 0
    
    # Recalculate spending after CryoSleep correction
    combined['TotalSpend'] = combined[spending_cols].sum(axis=1)
    combined['HasAnySpending'] = (combined['TotalSpend'] > 0).astype(int)
    
    print(f"✅ CryoSleep imputation complete")
    
    # STEP 4: Simple categorical imputation
    print("\n🌍 Categorical imputation...")
    
    # HomePlanet imputation
    def impute_homeplanet(val):
        val_str = str(val).lower()
        if val_str in ['earth', 'europa', 'mars']:
            return val_str.title()
        else:
            return 'Earth'  # Most common
    
    combined['HomePlanet'] = combined['HomePlanet'].apply(impute_homeplanet)
    
    # Destination imputation
    def impute_destination(val):
        val_str = str(val).lower()
        if 'trappist' in val_str:
            return 'TRAPPIST-1e'
        elif 'cancri' in val_str:
            return '55 Cancri e'
        elif 'pso' in val_str:
            return 'PSO J318.5-22'
        else:
            return 'TRAPPIST-1e'  # Most common
    
    combined['Destination'] = combined['Destination'].apply(impute_destination)
    
    # VIP imputation
    def impute_vip_simple(row):
        vip_val = str(row['VIP']).lower()
        if vip_val in ['true', '1', 'yes']:
            return 1
        elif vip_val in ['false', '0', 'no']:
            return 0
        elif row['TotalSpend'] > 1000:  # High spenders more likely VIP
            return 1
        else:
            return 0  # Default to not VIP
    
    combined['VIP'] = combined.apply(impute_vip_simple, axis=1)
    
    # STEP 5: Age imputation
    print("\n🎂 Age imputation...")
    def impute_age_simple(row):
        if pd.notna(row['Age']) and row['Age'] > 0:
            return row['Age']
        elif pd.notna(row['GroupAge_mean']):
            return row['GroupAge_mean']
        elif row['CryoSleep'] == 1:
            return 35.0  # CryoSleep passengers tend to be older
        else:
            return 29.0  # Overall median approximation
    
    combined['Age'] = combined.apply(impute_age_simple, axis=1)
    
    # STEP 6: Cabin processing
    print("\n🏨 Cabin processing...")
    
    # Parse cabin information
    def parse_cabin(cabin_str):
        cabin_str = str(cabin_str)
        if cabin_str == 'nan' or cabin_str == 'None' or len(cabin_str) < 3:
            return 'Unknown', 0, 'Unknown'
        
        parts = cabin_str.split('/')
        if len(parts) == 3:
            deck = parts[0] if parts[0] else 'Unknown'
            try:
                num = float(parts[1]) if parts[1] else 0
            except:
                num = 0
            side = parts[2] if parts[2] else 'Unknown'
            return deck, num, side
        else:
            return 'Unknown', 0, 'Unknown'
    
    cabin_parsed = combined['Cabin'].apply(parse_cabin)
    combined['CabinDeck'] = [x[0] for x in cabin_parsed]
    combined['CabinNum'] = [x[1] for x in cabin_parsed]
    combined['CabinSide'] = [x[2] for x in cabin_parsed]
    
    # Impute missing cabin information using patterns
    def impute_cabin_deck(row):
        if row['CabinDeck'] != 'Unknown':
            return row['CabinDeck']
        elif row['HomePlanet'] == 'Earth':
            return 'F'
        elif row['HomePlanet'] == 'Europa':
            return 'B'
        elif row['HomePlanet'] == 'Mars':
            return 'E'
        else:
            return 'G'
    
    combined['CabinDeck'] = combined.apply(impute_cabin_deck, axis=1)
    
    def impute_cabin_side(row):
        if row['CabinSide'] != 'Unknown':
            return row['CabinSide']
        elif row['VIP'] == 1:
            return 'P'  # VIP prefer Port
        else:
            return 'S'  # Regular on Starboard
    
    combined['CabinSide'] = combined.apply(impute_cabin_side, axis=1)
    
    # Impute cabin numbers
    deck_medians = combined[combined['CabinNum'] > 0].groupby('CabinDeck')['CabinNum'].median()
    
    def impute_cabin_num(row):
        if row['CabinNum'] > 0:
            return row['CabinNum']
        elif row['CabinDeck'] in deck_medians:
            return deck_medians[row['CabinDeck']]
        else:
            return 500  # Default
    
    combined['CabinNum'] = combined.apply(impute_cabin_num, axis=1)
    
    print("✅ All missing values imputed")
    
    # STEP 7: Feature engineering
    print("\n⚙️ Creating engineered features...")
    
    # Recalculate spending features
    combined['TotalSpend'] = combined[spending_cols].sum(axis=1)
    combined['SpendPerPerson'] = combined['TotalSpend'] / combined['GroupSize']
    combined['HasAnySpending'] = (combined['TotalSpend'] > 0).astype(int)
    
    # Advanced spending features
    combined['SpendingDiversity'] = (combined[spending_cols] > 0).sum(axis=1)
    combined['MaxSpendCategory'] = combined[spending_cols].idxmax(axis=1)
    combined['SpendingVariance'] = combined[spending_cols].var(axis=1).fillna(0)
    
    # Log transformed spending
    for col in spending_cols:
        combined[f'log_{col}'] = np.log1p(combined[col])
    combined['log_TotalSpend'] = np.log1p(combined['TotalSpend'])
    
    # Age-based features
    combined['AgeGroup'] = pd.cut(combined['Age'],
                                 bins=[0, 12, 18, 25, 35, 50, 65, 100],
                                 labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'MiddleAge', 'Senior', 'Elder'])
    combined['IsMinor'] = (combined['Age'] < 18).astype(int)
    combined['IsElderly'] = (combined['Age'] > 65).astype(int)
    
    # Group interaction features
    combined['AgeDeviation'] = abs(combined['Age'] - combined['GroupAge_mean'])
    combined['IsOldestInGroup'] = (combined['Age'] == combined['GroupAge_max']).astype(int)
    combined['IsYoungestInGroup'] = (combined['Age'] == combined['GroupAge_min']).astype(int)
    
    # Cabin features
    combined['CabinRegion'] = pd.cut(combined['CabinNum'], bins=10, labels=False)
    combined['DeckSide'] = combined['CabinDeck'] + '_' + combined['CabinSide']
    
    # Journey features
    combined['Journey'] = combined['HomePlanet'] + '_to_' + combined['Destination']
    
    # STEP 8: Critical interaction features
    print("\n🔗 Creating interaction features...")
    
    # Key interactions
    combined['CryoSpend'] = combined['CryoSleep'] * combined['TotalSpend']
    combined['VIPSpend'] = combined['VIP'] * combined['TotalSpend']
    combined['CryoAge'] = combined['CryoSleep'] * combined['Age']
    combined['VIPAge'] = combined['VIP'] * combined['Age']
    combined['AloneSpend'] = combined['IsAlone'] * combined['TotalSpend']
    
    # STEP 9: Simple frequency encoding
    print("\n📊 Creating frequency encodings...")
    
    freq_encode_cols = ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide', 'MaxSpendCategory', 'Journey']
    for col in freq_encode_cols:
        freq_map = combined[col].value_counts(normalize=True).to_dict()
        combined[f'{col}_freq'] = combined[col].map(freq_map).fillna(0)
    
    # STEP 10: Simple target encoding (fixed)
    print("\n🎯 Creating target encodings...")
    
    train_data = combined[combined['is_train']].copy()
    if not train_data['Transported'].isna().all():
        target_encode_cols = ['HomePlanet', 'Destination', 'CabinDeck', 'MaxSpendCategory']
        
        for col in target_encode_cols:
            # Convert to string to avoid categorical issues
            combined[col] = combined[col].astype(str)
            
            # Calculate target means with smoothing
            target_means = train_data.groupby(col)['Transported'].agg(['mean', 'count'])
            global_mean = train_data['Transported'].mean()
            
            # Apply smoothing
            smoothing = 10
            target_means['smoothed_mean'] = (
                target_means['count'] * target_means['mean'] + smoothing * global_mean
            ) / (target_means['count'] + smoothing)
            
            target_map = target_means['smoothed_mean'].to_dict()
            
            # Create new column instead of trying to modify existing one
            combined[f'{col}_target_encoded'] = combined[col].map(target_map).fillna(global_mean)
    
    # STEP 11: Polynomial features
    print("\n🔧 Creating polynomial features...")
    
    combined['Age_squared'] = combined['Age'] ** 2
    combined['TotalSpend_squared'] = combined['TotalSpend'] ** 2
    combined['GroupSize_squared'] = combined['GroupSize'] ** 2
    
    # Ratio features
    combined['AgeGroupSizeRatio'] = combined['Age'] / (combined['GroupSize'] + 1)
    combined['SpendAgeRatio'] = combined['TotalSpend'] / (combined['Age'] + 1)
    
    # STEP 12: Missing value flags (from original data)
    print("\n🏷️ Creating missing value flags...")
    
    # Read original data to check for missing values
    train_orig = pd.read_csv(train_path)
    test_orig = pd.read_csv(test_path)
    
    missing_flag_cols = ['Age', 'Cabin', 'HomePlanet', 'Destination', 'CryoSleep', 'VIP'] + spending_cols
    
    for col in missing_flag_cols:
        train_missing = train_orig[col].isna() if col in train_orig.columns else pd.Series([False] * len(train_orig))
        test_missing = test_orig[col].isna() if col in test_orig.columns else pd.Series([False] * len(test_orig))
        combined_missing = pd.concat([train_missing, test_missing], ignore_index=True)
        combined[f'{col}_was_missing'] = combined_missing.astype(int)
    
    # STEP 13: Final preprocessing
    print("\n📦 Final preprocessing...")
    
    # Split back into train and test
    final_train = combined[combined['is_train']].copy()
    final_test = combined[~combined['is_train']].copy()
    
    # Remove helper columns
    cols_to_remove = [
        'is_train', 'GroupId', 'PersonInGroup', 'Cabin', 'Name',
        'GroupAge_mean', 'GroupAge_std', 'GroupAge_min', 'GroupAge_max'
    ]
    
    for col in cols_to_remove:
        if col in final_train.columns:
            final_train = final_train.drop(columns=[col])
        if col in final_test.columns:
            final_test = final_test.drop(columns=[col])
    
    # Remove Transported from test set
    if 'Transported' in final_test.columns:
        final_test = final_test.drop(columns=['Transported'])
    
    # Convert categorical columns to numeric using simple label encoding
    categorical_cols = ['AgeGroup', 'MaxSpendCategory', 'Journey', 'DeckSide', 'CabinDeck', 'CabinSide', 'HomePlanet', 'Destination']
    
    for col in categorical_cols:
        if col in final_train.columns:
            # Create mapping from combined unique values
            combined_values = pd.concat([final_train[col], final_test[col]]).astype(str).unique()
            value_map = {val: idx for idx, val in enumerate(combined_values)}
            
            final_train[col] = final_train[col].astype(str).map(value_map)
            final_test[col] = final_test[col].astype(str).map(value_map)
    
    # Fill any remaining NaN values
    final_train = final_train.fillna(0)
    final_test = final_test.fillna(0)
    
    print(f"📊 Final shapes: Train {final_train.shape}, Test {final_test.shape}")
    print(f"📊 Total features: {final_train.shape[1] - 1}")
    
    # Save cleaned datasets
    final_train.to_csv('train_advanced_cleaned.csv', index=False)
    final_test.to_csv('test_advanced_cleaned.csv', index=False)
    
    print("💾 Saved: train_advanced_cleaned.csv, test_advanced_cleaned.csv")
    
    # Data quality check
    print(f"\n🔍 Data Quality Check:")
    print(f"Train missing values: {final_train.isnull().sum().sum()}")
    print(f"Test missing values: {final_test.isnull().sum().sum()}")
    print(f"Train data types: numeric={final_train.select_dtypes(include=[np.number]).shape[1]}")
    print(f"Test data types: numeric={final_test.select_dtypes(include=[np.number]).shape[1]}")
    
    print(f"\n💡 Key Improvements:")
    print("✅ Fixed categorical data type issues")
    print("✅ Robust missing value imputation")
    print("✅ CryoSleep-spending relationship exploited")
    print("✅ Group-based feature engineering")
    print("✅ Advanced interaction features")
    print(f"✅ {final_train.shape[1] - 1} total engineered features")
    
    return final_train, final_test

if __name__ == "__main__":
    print("🚀 Starting Fixed Advanced Data Cleaning...")
    train_clean, test_clean = advanced_data_cleaning()
    print("✅ Fixed advanced cleaning complete!")
    print(f"\n🎯 Expected Accuracy Improvement: +2-3% over baseline")
    print("Ready for advanced modeling pipeline!")
