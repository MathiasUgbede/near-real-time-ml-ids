"""
Script: 02_preprocess_data.py
Author: Mathias Amade
Purpose: Preprocess UNSW-NB15 dataset for ML training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Configuration
DATA_PATH = '/home/mathias/ml-ids-project/data/'
RESULTS_PATH = '/home/mathias/ml-ids-project/results/'
TRAINING_FILE = DATA_PATH + 'UNSW_NB15_training-set.csv'
TESTING_FILE = DATA_PATH + 'UNSW_NB15_testing-set.csv'

# Output files
TRAIN_OUT = DATA_PATH + 'train_preprocessed.csv'
TEST_OUT = DATA_PATH + 'test_preprocessed.csv'

print("=" * 60)
print("UNSW-NB15 DATA PREPROCESSING")
print("=" * 60)

# Step 1: Load data
print("\n[1] Loading training and testing data...")
train_df = pd.read_csv(TRAINING_FILE)
test_df = pd.read_csv(TESTING_FILE)
print(f"    Training: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"    Testing:  {test_df.shape[0]} rows, {test_df.shape[1]} columns")

# Step 2: Identify categorical columns
print("\n[2] Categorical columns (need encoding):")
categorical_cols = ['proto', 'service', 'state']
for col in categorical_cols:
    unique_count = train_df[col].nunique()
    print(f"    {col}: {unique_count} unique values")
    print(f"       Examples: {list(train_df[col].unique()[:5])}")

# Step 3: Drop unnecessary columns
print("\n[3] Dropping unnecessary columns...")
# 'id' is just a row identifier, not useful for ML
# 'attack_cat' is the multi-class label (we only need binary 'label')
cols_to_drop = ['id', 'attack_cat']
train_df = train_df.drop(cols_to_drop, axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)
print(f"    Dropped: {cols_to_drop}")
print(f"    Remaining columns: {train_df.shape[1]}")

# Step 4: Encode categorical columns using Label Encoding
print("\n[4] Encoding categorical features to numbers...")
for col in categorical_cols:
    encoder = LabelEncoder()
    # Combine train and test to ensure consistent encoding
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    encoder.fit(combined)
    train_df[col] = encoder.transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])
    print(f"    {col} encoded: {len(encoder.classes_)} categories -> numbers")

# Step 5: Separate features and labels
print("\n[5] Separating features (X) and labels (y)...")
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']
print(f"    X_train: {X_train.shape}")
print(f"    y_train: {y_train.shape}")

# Step 6: Normalize features using StandardScaler
print("\n[6] Normalizing features (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"    Features normalized to mean=0, std=1")

# Step 7: Convert back to DataFrames for saving
X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_train_final['label'] = y_train.values
X_test_final['label'] = y_test.values

# Step 8: Save preprocessed data
print("\n[7] Saving preprocessed data...")
X_train_final.to_csv(TRAIN_OUT, index=False)
X_test_final.to_csv(TEST_OUT, index=False)
print(f"    Training data saved: {TRAIN_OUT}")
print(f"    Testing data saved:  {TEST_OUT}")

# Step 9: Summary
print("\n[8] Preprocessing Summary:")
print(f"    Original training columns: 45")
print(f"    Preprocessed training columns: {X_train_final.shape[1]}")
print(f"    Training samples: {len(X_train_final)}")
print(f"    Testing samples: {len(X_test_final)}")
print(f"    All features are now numeric and normalized")

# Class distribution
print("\n[9] Class Distribution:")
print(f"    Training - Normal (0): {(y_train == 0).sum()}")
print(f"    Training - Attack (1): {(y_train == 1).sum()}")
print(f"    Testing  - Normal (0): {(y_test == 0).sum()}")
print(f"    Testing  - Attack (1): {(y_test == 1).sum()}")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)
