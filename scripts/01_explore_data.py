"""
Script: 01_explore_data.py
Purpose: Load and explore the UNSW-NB15 dataset
Author: Mathias Amade
Project: MSc Dissertation - ML-Based IDS
"""

import pandas as pd
import numpy as np

# Configuration
DATA_PATH = '/home/mathias/ml-ids-project/data/'
TRAINING_FILE = DATA_PATH + 'UNSW_NB15_training-set.csv'
TESTING_FILE = DATA_PATH + 'UNSW_NB15_testing-set.csv'

print("=" * 60)
print("UNSW-NB15 DATASET EXPLORATION")
print("=" * 60)

# Load training data
print("\n[1] Loading training data...")
train_df = pd.read_csv(TRAINING_FILE)
print(f"    Training set loaded: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

# Load testing data
print("\n[2] Loading testing data...")
test_df = pd.read_csv(TESTING_FILE)
print(f"    Testing set loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

# Show column names
print("\n[3] Features in dataset:")
print(train_df.columns.tolist())

# Show first 5 rows
print("\n[4] First 5 rows of training data:")
print(train_df.head())

# Check label distribution
print("\n[5] Label distribution in TRAINING set:")
print(f"    Normal (0): {(train_df['label']==0).sum()}")
print(f"    Attack (1): {(train_df['label']==1).sum()}")

# Check attack categories
print("\n[6] Attack categories distribution:")
print(train_df['attack_cat'].value_counts())

# Check for missing values
print("\n[7] Missing values check:")
missing = train_df.isnull().sum().sum()
print(f"    Total missing values: {missing}")

# Data types
print("\n[8] Data types summary:")
print(train_df.dtypes.value_counts())

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE!")
print("=" * 60)
