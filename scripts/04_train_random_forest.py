"""
Script: 04_train_random_forest.py
Author: Mathias Amade
Purpose: Train and evaluate Random Forest on UNSW-NB15
"""

import pandas as pd
import numpy as np
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = '/home/mathias/ml-ids-project/data/'
MODELS_PATH = '/home/mathias/ml-ids-project/models/'
RESULTS_PATH = '/home/mathias/ml-ids-project/results/'

# Preprocessed and feature-selected UNSW-NB15 data
TRAIN_FILE = DATA_PATH + 'train_selected.csv'
TEST_FILE = DATA_PATH + 'test_selected.csv'


print("=" * 60)
print("RANDOM FOREST CLASSIFIER - TRAINING & EVALUATION")
print("=" * 60)


# ============================================================
# 2. LOAD PREPARED TRAINING AND TEST DATA
# ============================================================

print("\n[1] Loading feature-selected data...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"    Training: {train_df.shape}")
print(f"    Testing:  {test_df.shape}")


# ============================================================
# 3. SEPARATE FEATURES FROM TARGET LABEL
# ============================================================

print("\n[2] Preparing features and labels...")

# Predictive features used by the model
X_train = train_df.drop('label', axis=1)
X_test = test_df.drop('label', axis=1)

# Binary target:
# 0 = Normal
# 1 = Attack
y_train = train_df['label']
y_test = test_df['label']

print(f"    Features: {X_train.shape[1]}")
print(f"    Training samples: {len(X_train)}")
print(f"    Testing samples: {len(X_test)}")


# ============================================================
# 4. CONFIGURE RANDOM FOREST
# ============================================================

print("\n[3] Configuring Random Forest classifier...")

rf_model = RandomForestClassifier(
    n_estimators=100,        # 100 decision trees
    max_depth=20,            # Limit tree depth
    min_samples_split=5,     # Minimum samples needed to split a node
    min_samples_leaf=2,      # Minimum samples in each leaf
    max_features='sqrt',     # Features considered at each split
    random_state=42,         # Reproducible results
    n_jobs=-1                # Use all available CPU cores
)

print("    n_estimators: 100")
print("    max_depth: 20")
print("    min_samples_split: 5")
print("    min_samples_leaf: 2")


# ============================================================
# 5. TRAIN RANDOM FOREST
# ============================================================

print("\n[4] Training Random Forest model...")

# Measure training time
train_start = time.time()

# Learn patterns from training features and known labels
rf_model.fit(X_train, y_train)

train_time = time.time() - train_start

print(f"    Training complete in {train_time:.2f} seconds")


# ============================================================
# 6. TEST THE TRAINED MODEL
# ============================================================

print("\n[5] Making predictions on test set...")

predict_start = time.time()

# Predict binary labels for unseen test records
y_pred = rf_model.predict(X_test)

predict_time = time.time() - predict_start

print(f"    Predictions complete in {predict_time:.2f} seconds")

# Average batch prediction time per sample
per_sample_time = (predict_time / len(X_test)) * 1000

print(f"    Average prediction time: {per_sample_time:.4f} ms per sample")


# ============================================================
# 7. CALCULATE CLASSIFICATION METRICS
# ============================================================

print("\n[6] Calculating performance metrics...")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Probability of the positive/attack class used for ROC AUC
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)

print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"    Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"    Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"    F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"    ROC AUC:   {auc:.4f}")


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

print("\n[7] Confusion Matrix:")

cm = confusion_matrix(y_test, y_pred)

# TN = Normal correctly identified
# FP = Normal incorrectly flagged as attack
# FN = Attack missed by the IDS
# TP = Attack correctly detected
print(f"    True Negatives:  {cm[0][0]}")
print(f"    False Positives: {cm[0][1]}")
print(f"    False Negatives: {cm[1][0]}")
print(f"    True Positives:  {cm[1][1]}")


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\n[8] Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=['Normal', 'Attack']
    )
)


# ============================================================
# 10. SAVE CONFUSION MATRIX IMAGE
# ============================================================

print("\n[9] Creating confusion matrix visualization...")

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Normal', 'Attack'],
    yticklabels=['Normal', 'Attack']
)

plt.title('Random Forest - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()

cm_file = RESULTS_PATH + 'rf_confusion_matrix.png'

plt.savefig(
    cm_file,
    dpi=100,
    bbox_inches='tight'
)

print(f"    Confusion matrix saved: {cm_file}")


# ============================================================
# 11. SAVE TRAINED RANDOM FOREST MODEL
# ============================================================

print("\n[10] Saving trained model...")

model_file = MODELS_PATH + 'random_forest_model.pkl'

# Save trained model so VM2 can load it later
# without retraining during the streamed experiment
joblib.dump(rf_model, model_file)

print(f"    Model saved: {model_file}")


# ============================================================
# 12. SAVE RESULTS SUMMARY
# ============================================================

print("\n[11] Saving results summary...")

results_file = RESULTS_PATH + 'random_forest_results.txt'

with open(results_file, 'w') as f:

    f.write("RANDOM FOREST CLASSIFIER RESULTS\n")
    f.write("=" * 40 + "\n")

    f.write(f"Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision:  {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall:     {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"F1-Score:   {f1:.4f} ({f1*100:.2f}%)\n")
    f.write(f"ROC AUC:    {auc:.4f}\n")

    f.write(f"\nTraining Time:   {train_time:.2f} seconds\n")
    f.write(f"Prediction Time: {predict_time:.2f} seconds\n")
    f.write(f"Per Sample:      {per_sample_time:.4f} ms\n")

    f.write("\nConfusion Matrix:\n")
    f.write("                Predicted\n")
    f.write("                Normal   Attack\n")
    f.write(f"Actual Normal:  {cm[0][0]:>6}  {cm[0][1]:>6}\n")
    f.write(f"Actual Attack:  {cm[1][0]:>6}  {cm[1][1]:>6}\n")

print(f"    Results saved: {results_file}")


print("\n" + "=" * 60)
print("RANDOM FOREST TRAINING COMPLETE!")
print("=" * 60)
