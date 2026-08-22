"""
Script: 03_feature_selection.py
Author: Mathias Amade
Purpose: Select most important features using Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = '/home/mathias/ml-ids-project/data/'
RESULTS_PATH = '/home/mathias/ml-ids-project/results/'
TRAIN_FILE = DATA_PATH + 'train_preprocessed.csv'
TEST_FILE = DATA_PATH + 'test_preprocessed.csv'

# Number of top features to select
TOP_N_FEATURES = 20

print("=" * 60)
print("FEATURE SELECTION - Random Forest Importance")
print("=" * 60)

# Step 1: Load preprocessed data
print("\n[1] Loading preprocessed data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
print(f"    Training: {train_df.shape}")
print(f"    Testing:  {test_df.shape}")

# Step 2: Separate features and labels
print("\n[2] Separating features and labels...")
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']
print(f"    Features: {X_train.shape[1]}")

# Step 3: Train Random Forest for feature importance
print("\n[3] Training Random Forest to calculate feature importance...")
print("    (This may take 2-3 minutes...)")
rf_selector = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
rf_selector.fit(X_train, y_train)
print("    Training complete!")

# Step 4: Get feature importance scores
print("\n[4] Calculating feature importance...")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_selector.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n    Top 10 most important features:")
print("    " + "-" * 50)
for idx, row in feature_importance.head(10).iterrows():
    print(f"    {row['feature']:30s} {row['importance']:.4f}")

# Step 5: Select top N features
print(f"\n[5] Selecting top {TOP_N_FEATURES} features...")
top_features = feature_importance.head(TOP_N_FEATURES)['feature'].tolist()
print(f"    Selected {len(top_features)} features")

# Step 6: Create reduced datasets
print("\n[6] Creating reduced datasets...")
X_train_selected = X_train[top_features].copy()
X_test_selected = X_test[top_features].copy()
X_train_selected['label'] = y_train.values
X_test_selected['label'] = y_test.values

# Step 7: Save selected features data
print("\n[7] Saving reduced datasets...")
TRAIN_OUT = DATA_PATH + 'train_selected.csv'
TEST_OUT = DATA_PATH + 'test_selected.csv'
X_train_selected.to_csv(TRAIN_OUT, index=False)
X_test_selected.to_csv(TEST_OUT, index=False)
print(f"    Saved: {TRAIN_OUT}")
print(f"    Saved: {TEST_OUT}")

# Step 8: Save feature importance to CSV
importance_file = RESULTS_PATH + 'feature_importance.csv'
feature_importance.to_csv(importance_file, index=False)
print(f"    Saved: {importance_file}")

# Step 9: Create bar chart visualization
print("\n[8] Creating feature importance chart...")
plt.figure(figsize=(10, 8))
top20 = feature_importance.head(20)
plt.barh(range(len(top20)), top20['importance'], color='steelblue')
plt.yticks(range(len(top20)), top20['feature'])
plt.xlabel('Feature Importance Score')
plt.title('Top 20 Most Important Features (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
chart_file = RESULTS_PATH + 'feature_importance_chart.png'
plt.savefig(chart_file, dpi=100, bbox_inches='tight')
print(f"    Chart saved: {chart_file}")

# Step 10: Summary
print("\n[9] Feature Selection Summary:")
print(f"    Original features: {X_train.shape[1]}")
print(f"    Selected features: {len(top_features)}")
print(f"    Reduction: {X_train.shape[1] - len(top_features)} features removed")
print(f"    Training samples: {len(X_train_selected)}")
print(f"    Testing samples: {len(X_test_selected)}")

print("\n" + "=" * 60)
print("FEATURE SELECTION COMPLETE!")
print("=" * 60)
