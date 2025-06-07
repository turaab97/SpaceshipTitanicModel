import pandas as pd

# Read the working submission to get correct PassengerIds
minimal_sub = pd.read_csv('submission.csv')
print(f"Minimal submission shape: {minimal_sub.shape}")
print("Sample PassengerIds:", minimal_sub['PassengerId'].head().tolist())

# Read the broken submission to get the new predictions
broken_sub = pd.read_csv('submission_kmeans_stacking3.csv')
print(f"Broken submission shape: {broken_sub.shape}")
print("Sample predictions:", broken_sub['Transported'].head().tolist())

# Create fixed submission by combining correct PassengerIds with new predictions
fixed_submission = pd.DataFrame({
    'PassengerId': minimal_sub['PassengerId'],
    'Transported': broken_sub['Transported']
})

print(f"Fixed submission shape: {fixed_submission.shape}")
print("Fixed submission sample:")
print(fixed_submission.head())

# Save the corrected submission
fixed_submission.to_csv('submission_kmeans_stacking3_fixed.csv', index=False)
print("✅ Created submission_kmeans_stacking3_fixed.csv with correct PassengerId format")

# Verify the fix
verification = pd.read_csv('submission_kmeans_stacking3_fixed.csv')
print(f"\nVerification:")
print(f"Shape: {verification.shape}")
print(f"PassengerId dtype: {verification['PassengerId'].dtype}")
print(f"Sample PassengerIds: {verification['PassengerId'].head().tolist()}")
print(f"Transported values: {verification['Transported'].value_counts().to_dict()}")
