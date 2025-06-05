import pandas as pd
import gender_guesser.detector as gender

# 0. pip install gender_guesser

# 1. Load and coerce Name → str
data = pd.read_csv('/Users/syedturab/Downloads/mmai-2-checklist/train.csv')
data['Name'] = data['Name'].fillna('').astype(str)

# 2. Extract first name, coerce to str
data['FirstName'] = data['Name'].str.split().str[0]
data['FirstName'] = data['FirstName'].fillna('').astype(str)

# 3. Init detector
detector = gender.Detector(case_sensitive=False)

# 4. Safe gender inference
def safe_gender(name):
    # name guaranteed to be a string
    name = name.strip()
    if not name:
        return 'unknown'
    return detector.get_gender(name)

data['GenderRaw'] = data['FirstName'].apply(safe_gender)

# 5. Map to Male/Female/Unknown
def map_gender(g):
    if g in ('male','mostly_male'):
        return 'Male'
    if g in ('female','mostly_female'):
        return 'Female'
    return 'Unknown'

data['Gender'] = data['GenderRaw'].apply(map_gender)

# 6. Write out the new CSV
output_path = '/Users/syedturab/Desktop/Queens MMAI Course Material/MMAI 869/Group Project /trainname_with_gender.csv'
data.to_csv(output_path, index=False)

print(f"Done — new file at {output_path}")
