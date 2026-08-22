"""
Script: 05_train_gradient_boosting.py
Author: Mathias Amade
Purpose: Train Gradient Boosting classifier on UNSW-NB15 dataset
"""

import pandas as pd
import numpy as np
import time
import joblib
from sklearn.ensemble import GradientBoostingClassifier
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
print("GRADIENT BOOSTING CLASSIFIER - TRAINING & EVALUATION")
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

# Step 3: Configure Gradient Boosting
print("\n[3] Configuring Gradient Boosting classifier...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
print("    n_estimators: 100")
print("    learning_rate: 0.1")
print("    max_depth: 5")
print("    min_samples_split: 5")
print("    min_samples_leaf: 2")

# Step 4: Train the model
print("\n[4] Training Gradient Boosting model...")
print("    (This is slower than Random Forest - may take 5-10 minutes...)")
train_start = time.time()
gb_model.fit(X_train, y_train)
train_time = time.time() - train_start
print(f"    Training complete in {train_time:.2f} seconds")

# Step 5: Make predictions
print("\n[5] Making predictions on test set...")
predict_start = time.time()
y_pred = gb_model.predict(X_test)
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

y_pred_proba = gb_model.predict_proba(X_test)[:, 1]
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Gradient Boosting - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
cm_file = RESULTS_PATH + 'gb_confusion_matrix.png'
plt.savefig(cm_file, dpi=100, bbox_inches='tight')
print(f"    Confusion matrix saved: {cm_file}")

# Step 10: Save trained model
print("\n[10] Saving trained model...")
model_file = MODELS_PATH + 'gradient_boosting_model.pkl'
joblib.dump(gb_model, model_file)
print(f"    Model saved: {model_file}")

# Step 11: Save results to file
print("\n[11] Saving results summary...")
results_file = RESULTS_PATH + 'gradient_boosting_results.txt'
with open(results_file, 'w') as f:
    f.write("GRADIENT BOOSTING CLASSIFIER RESULTS\n")
    f.write("=" * 40 + "\n")
    f.write(f"Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision:  {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall:     {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"F1-Score:   {f1:.4f} ({f1*100:.2f}%)\n")
    f.write(f"ROC AUC:    {auc:.4f}\n")
    f.write(f"\nTraining Time:   {train_time:.2f} seconds\n")
    f.write(f"Prediction Time: {predict_time:.2f} seconds\n")
    f.write(f"Per Sample:      {per_sample_time:.4f} ms\n")
    f.write(f"\nConfusion Matrix:\n")
    f.write(f"                Predicted\n")
    f.write(f"                Normal   Attack\n")
    f.write(f"Actual Normal:  {cm[0][0]:>6}  {cm[0][1]:>6}\n")
    f.write(f"Actual Attack:  {cm[1][0]:>6}  {cm[1][1]:>6}\n")
print(f"    Results saved: {results_file}")

print("\n" + "=" * 60)
print("GRADIENT BOOSTING TRAINING COMPLETE!")
print("=" * 60)
