"""
Script: 06b_svm_save_results.py
Purpose: Rebuild SVM confusion matrix + results file from known metrics
         (avoids retraining - the original run's metrics are used directly)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

RESULTS_PATH = '/home/mathias/ml-ids-project/results/'

# Metrics captured from the completed SVM training run
accuracy = 0.8793
precision = 0.9813
recall = 0.8386
f1 = 0.9044
auc = 0.9593
train_time = 800.67
predict_time = 158.11
per_sample_time = 0.9017

# Confusion matrix values from the run
# [[TN, FP], [FN, TP]]
cm = np.array([[54094, 1906],
               [19262, 100079]])

print("Recreating SVM confusion matrix image...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Support Vector Machine - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
cm_file = RESULTS_PATH + 'svm_confusion_matrix.png'
plt.savefig(cm_file, dpi=100, bbox_inches='tight')
print(f"    Saved: {cm_file}")

print("Writing SVM results summary...")
results_file = RESULTS_PATH + 'svm_results.txt'
with open(results_file, 'w') as f:
    f.write("SUPPORT VECTOR MACHINE CLASSIFIER RESULTS\n")
    f.write("=" * 40 + "\n")
    f.write(f"Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision:  {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall:     {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"F1-Score:   {f1:.4f} ({f1*100:.2f}%)\n")
    f.write(f"ROC AUC:    {auc:.4f}\n")
    f.write(f"\nTraining Time:   {train_time:.2f} seconds ({train_time/60:.1f} minutes)\n")
    f.write(f"Prediction Time: {predict_time:.2f} seconds\n")
    f.write(f"Per Sample:      {per_sample_time:.4f} ms\n")
    f.write(f"\nConfusion Matrix:\n")
    f.write(f"                Predicted\n")
    f.write(f"                Normal   Attack\n")
    f.write(f"Actual Normal:  {cm[0][0]:>6}  {cm[0][1]:>6}\n")
    f.write(f"Actual Attack:  {cm[1][0]:>6}  {cm[1][1]:>6}\n")
print(f"    Saved: {results_file}")
print("Done.")
