import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Example: Credit risk classification
# Let's simulate some data
np.random.seed(42)
n_samples = 1000

# Simulate features (income, credit_score, debt_ratio, etc.)
X = np.random.randn(n_samples, 5)
# Simulate target: 1 = risky, 0 = not risky
y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.3 > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get prediction probabilities
y_proba = model.predict_proba(X_test)[:, 1]

def evaluate_threshold(y_true, y_proba, threshold):
    """Evaluate model performance at a given threshold"""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'threshold': threshold,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'fp_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
    }

# Test different thresholds
thresholds = np.arange(0.1, 0.9, 0.05)
results = []

for threshold in thresholds:
    result = evaluate_threshold(y_test, y_proba, threshold)
    results.append(result)

results_df = pd.DataFrame(results)

# Find optimal threshold for minimizing FP
min_fp_idx = results_df['fp'].idxmin()
optimal_threshold = results_df.loc[min_fp_idx, 'threshold']

print(f"Optimal threshold to minimize FP: {optimal_threshold:.3f}")
print(f"At this threshold:")
print(f"  FP: {results_df.loc[min_fp_idx, 'fp']}")
print(f"  FN: {results_df.loc[min_fp_idx, 'fn']}")
print(f"  Precision: {results_df.loc[min_fp_idx, 'precision']:.3f}")
print(f"  Recall: {results_df.loc[min_fp_idx, 'recall']:.3f}")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: FP vs Threshold
axes[0, 0].plot(results_df['threshold'], results_df['fp'], 'b-', linewidth=2)
axes[0, 0].axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
axes[0, 0].set_xlabel('Threshold')
axes[0, 0].set_ylabel('False Positives (FP)')
axes[0, 0].set_title('FP vs Threshold')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: Precision vs Recall
axes[0, 1].plot(results_df['recall'], results_df['precision'], 'g-', linewidth=2)
axes[0, 1].scatter(results_df.loc[min_fp_idx, 'recall'], results_df.loc[min_fp_idx, 'precision'], 
                   color='red', s=100, zorder=5, label=f'Min FP threshold')
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curve')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot 3: FP vs FN trade-off
axes[1, 0].plot(results_df['fp'], results_df['fn'], 'purple', linewidth=2)
axes[1, 0].scatter(results_df.loc[min_fp_idx, 'fp'], results_df.loc[min_fp_idx, 'fn'], 
                   color='red', s=100, zorder=5, label=f'Min FP threshold')
axes[1, 0].set_xlabel('False Positives (FP)')
axes[1, 0].set_ylabel('False Negatives (FN)')
axes[1, 0].set_title('FP vs FN Trade-off')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot 4: All metrics vs threshold
axes[1, 1].plot(results_df['threshold'], results_df['precision'], 'b-', label='Precision')
axes[1, 1].plot(results_df['threshold'], results_df['recall'], 'g-', label='Recall')
axes[1, 1].plot(results_df['threshold'], results_df['f1'], 'orange', label='F1-Score')
axes[1, 1].axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Metrics vs Threshold')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Show detailed results table
print("\nDetailed results at different thresholds:")
print(results_df.round(3).to_string(index=False)) 