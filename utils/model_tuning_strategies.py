import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Simulate credit risk data
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 5)
y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.3 > 0).astype(int)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate model and return confusion matrix"""
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    }

print("=== Strategy 1: Class Weights ===")
# Method 1: Use class weights to penalize FP more heavily
# Higher weight for negative class (not risky) = more penalty for FP
class_weights = {0: 2.0, 1: 1.0}  # Penalize FP more than FN

rf_weighted = RandomForestClassifier(
    n_estimators=100, 
    class_weight=class_weights,
    random_state=42
)
rf_weighted.fit(X_train, y_train)

results_weighted = evaluate_model(rf_weighted, X_test, y_test)
print(f"With class weights (penalizing FP):")
print(f"  FP: {results_weighted['fp']}")
print(f"  FN: {results_weighted['fn']}")
print(f"  Precision: {results_weighted['precision']:.3f}")
print(f"  Recall: {results_weighted['recall']:.3f}")

print("\n=== Strategy 2: Cost-Sensitive Learning ===")
# Method 2: Use cost matrix in training
# Define cost matrix: [TN_cost, FP_cost, FN_cost, TP_cost]
# Higher cost for FP means model will try harder to avoid it
cost_matrix = np.array([[0, 5, 1, 0]])  # FP cost = 5, FN cost = 1

# For Random Forest, we can use sample_weight
sample_weights = np.ones(len(y_train))
for i, label in enumerate(y_train):
    if label == 0:  # Negative class (not risky)
        sample_weights[i] = 2.0  # Higher weight = more important to classify correctly

rf_cost_sensitive = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cost_sensitive.fit(X_train, y_train, sample_weight=sample_weights)

results_cost = evaluate_model(rf_cost_sensitive, X_test, y_test)
print(f"With cost-sensitive learning:")
print(f"  FP: {results_cost['fp']}")
print(f"  FN: {results_cost['fn']}")
print(f"  Precision: {results_cost['precision']:.3f}")
print(f"  Recall: {results_cost['recall']:.3f}")

print("\n=== Strategy 3: Ensemble with Voting ===")
# Method 3: Use ensemble methods that are more conservative
from sklearn.ensemble import VotingClassifier

# Train multiple models
rf1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf2 = RandomForestClassifier(n_estimators=100, random_state=43)
rf3 = RandomForestClassifier(n_estimators=100, random_state=44)

# Use voting='hard' with majority voting (more conservative)
ensemble = VotingClassifier(
    estimators=[('rf1', rf1), ('rf2', rf2), ('rf3', rf3)],
    voting='hard'  # Requires majority agreement
)
ensemble.fit(X_train, y_train)

results_ensemble = evaluate_model(ensemble, X_test, y_test)
print(f"With ensemble voting:")
print(f"  FP: {results_ensemble['fp']}")
print(f"  FN: {results_ensemble['fn']}")
print(f"  Precision: {results_ensemble['precision']:.3f}")
print(f"  Recall: {results_ensemble['recall']:.3f}")

print("\n=== Strategy 4: Hyperparameter Tuning for FP Minimization ===")
# Method 4: Grid search with custom scoring
from sklearn.metrics import make_scorer

def fp_scorer(y_true, y_pred):
    """Custom scorer that minimizes FP"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return -fp  # Negative because we want to minimize FP

# Create custom scorer
fp_scorer_obj = make_scorer(fp_scorer, greater_is_better=False)

# Grid search with custom scorer
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': [None, {0: 1.5, 1: 1}, {0: 2, 1: 1}]
}

rf_tuned = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf_tuned, 
    param_grid, 
    scoring=fp_scorer_obj,
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters for minimizing FP: {grid_search.best_params_}")
results_tuned = evaluate_model(grid_search.best_estimator_, X_test, y_test)
print(f"With tuned hyperparameters:")
print(f"  FP: {results_tuned['fp']}")
print(f"  FN: {results_tuned['fn']}")
print(f"  Precision: {results_tuned['precision']:.3f}")
print(f"  Recall: {results_tuned['recall']:.3f}")

print("\n=== Strategy 5: Two-Stage Classification ===")
# Method 5: Use a two-stage approach
# Stage 1: High-recall model to catch most risky cases
# Stage 2: High-precision model to filter out FP from stage 1

# Stage 1: Train model with high recall (low threshold)
rf_stage1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_stage1.fit(X_train, y_train)

# Get high-recall predictions (low threshold = 0.3)
y_proba_stage1 = rf_stage1.predict_proba(X_test)[:, 1]
y_pred_stage1 = (y_proba_stage1 >= 0.3).astype(int)  # Low threshold for high recall

# Stage 2: Only apply second model to cases flagged as risky by stage 1
# In practice, you'd train a second model on the subset of data
# For demonstration, we'll use a more conservative threshold on the same model
y_pred_stage2 = (y_proba_stage1 >= 0.7).astype(int)  # High threshold for high precision

# Combine results (only flag as risky if both stages agree)
y_pred_two_stage = ((y_proba_stage1 >= 0.3) & (y_proba_stage1 >= 0.7)).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_two_stage).ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"With two-stage classification:")
print(f"  FP: {fp}")
print(f"  FN: {fn}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")

# Summary comparison
print("\n=== Summary Comparison ===")
methods = {
    'Baseline': evaluate_model(RandomForestClassifier(random_state=42).fit(X_train, y_train), X_test, y_test),
    'Class Weights': results_weighted,
    'Cost Sensitive': results_cost,
    'Ensemble': results_ensemble,
    'Tuned': results_tuned,
    'Two-Stage': {'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall}
}

summary_df = pd.DataFrame(methods).T
print(summary_df[['fp', 'fn', 'precision', 'recall']].round(3)) 