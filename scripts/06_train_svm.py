"""
Script: 06_train_svm.py
Author: Mathias Amade
Purpose: Train Support Vector Machine classifier on UNSW-NB15 dataset
"""

import pandas as pd
import numpy as np
import time
import joblib
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH = '/home/mathias/ml-ids-project/data/'
MODELS_PATH = '/home/mathias/ml-ids-project/models/'
RESULTS_PATH = '/home/mathias/ml-ids-project/results/'
TRAIN_FILE = DATA_PATH + 'train_selected.csv'
TEST_FILE = DATA_PATH + 'test_selected.csv'

print("=" * 60)
print("SUPPORT VECTOR MACHINE CLASSIFIER - TRAINING & EVALUATION")
print("=" * 60)

# Step 1: Load data
print("\n[1] Loading feature-selected data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
print(f"    Training: {train_df.shape}")
print(f"    Testing:  {test_df.shape}")

# Step 2: Separate features and labels
print("\n[2] Preparing features and labels...")
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']
print(f"    Features: {X_train.shape[1]}")
print(f"    Training samples: {len(X_train)}")
print(f"    Testing samples: {len(X_test)}")

# Step 3: Configure SVM
print("\n[3] Configuring Support Vector Machine classifier...")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    cache_size=500,
    random_state=42
)
print("    kernel: rbf (radial basis function)")
print("    C: 1.0")
print("    gamma: scale")
print("    probability: True")

# Step 4: Train the model
print("\n[4] Training SVM model...")
print("    (SVM is the slowest algorithm - this may take 20-40 minutes...)")
print("    (Please be patient and do not close the terminal)")
train_start = time.time()
svm_model.fit(X_train, y_train)
train_time = time.time() - train_start
print(f"    Training complete in {train_time:.2f} seconds ({train_time/60:.1f} minutes)")

# Step 5: Make predictions
print("\n[5] Making predictions on test set...")
predict_start = time.time()
y_pred = svm_model.predict(X_test)
predict_time = time.time() - predict_start
print(f"    Predictions complete in {predict_time:.2f} seconds")

per_sample_time = (predict_time / len(X_test)) * 1000
print(f"    Average prediction time: {per_sample_time:.4f} ms per sample")

# Step 6: Calculate metrics
print("\n[6] Calculating performance metrics...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

y_pred_proba = svm_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"    Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"    Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"    F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"    ROC AUC:   {auc:.4f}")

# Step 7: Confusion Matrix
print("\n[7] Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"    True Negatives (Normal correctly identified): {cm[0][0]}")
print(f"    False Positives (Normal misclassified as Attack): {cm[0][1]}")
print(f"    False Negatives (Attack missed): {cm[1][0]}")
print(f"    True Positives (Attack correctly detected): {cm[1][1]}")

# Step 8: Classification Report
print("\n[8] Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# Step 9: Visualize Confusion Matrix
print("\n[9] Creating confusion matrix visualization...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Support Vector Machine - Confusion Matrix')
plt.ylabel('Actual')
