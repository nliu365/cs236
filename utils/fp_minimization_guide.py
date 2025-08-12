"""
COMPREHENSIVE GUIDE: Minimizing False Positives (FP) in Production

This guide provides practical strategies for reducing FP in credit risk classification
and similar binary classification problems where FP is costly.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: UNDERSTAND YOUR BUSINESS CONTEXT
# ============================================================================

def analyze_business_costs():
    """
    Calculate the business impact of FP vs FN
    """
    # Example: Credit risk scenario
    fp_cost = 1000  # Cost of denying a good customer
    fn_cost = 5000  # Cost of approving a bad customer
    
    print("=== Business Cost Analysis ===")
    print(f"FP Cost (denying good customer): ${fp_cost}")
    print(f"FN Cost (approving bad customer): ${fn_cost}")
    print(f"Cost ratio (FN/FP): {fn_cost/fp_cost:.1f}")
    
    # If FN cost >> FP cost, you might want to minimize FN instead
    if fn_cost > fp_cost * 3:
        print("WARNING: FN cost is much higher than FP cost!")
        print("Consider if you really want to minimize FP")
    
    return fp_cost, fn_cost

# ============================================================================
# STEP 2: DATA EXPLORATION AND FP ANALYSIS
# ============================================================================

def analyze_fp_patterns(X, y, model):
    """
    Analyze where FP occur to understand patterns
    """
    y_pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # Find FP cases
    fp_indices = (y == 0) & (y_pred == 1)
    fp_data = X[fp_indices]
    
    print(f"\n=== FP Analysis ===")
    print(f"Total FP: {fp}")
    print(f"FP Rate: {fp/(fp+tn):.3f}")
    
    if len(fp_data) > 0:
        print(f"\nFP Cases Summary:")
        print(fp_data.describe())
        
        # Check if FP cases have specific patterns
        for col in fp_data.columns:
            if fp_data[col].std() < fp_data[col].mean() * 0.1:
                print(f"FP cases have low variance in {col}")
    
    return fp_data

# ============================================================================
# STEP 3: THRESHOLD OPTIMIZATION
# ============================================================================

def optimize_threshold_for_fp(y_true, y_proba, fp_weight=2.0):
    """
    Find optimal threshold that minimizes FP with custom weighting
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Custom scoring: penalize FP more heavily
        score = -(fp * fp_weight + fn)  # Negative because we minimize
        
        results.append({
            'threshold': threshold,
            'fp': fp, 'fn': fn, 'tp': tp, 'tn': tn,
            'score': score,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        })
    
    results_df = pd.DataFrame(results)
    optimal_idx = results_df['score'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    
    print(f"\n=== Threshold Optimization ===")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"At optimal threshold:")
    print(f"  FP: {results_df.loc[optimal_idx, 'fp']}")
    print(f"  FN: {results_df.loc[optimal_idx, 'fn']}")
    print(f"  Precision: {results_df.loc[optimal_idx, 'precision']:.3f}")
    print(f"  Recall: {results_df.loc[optimal_idx, 'recall']:.3f}")
    
    return optimal_threshold, results_df

# ============================================================================
# STEP 4: MODEL SELECTION FOR FP MINIMIZATION
# ============================================================================

def compare_models_for_fp(X_train, X_test, y_train, y_test):
    """
    Compare different models for FP minimization
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train with class weights to penalize FP
        if hasattr(model, 'class_weight'):
            model.set_params(class_weight={0: 2.0, 1: 1.0})
        
        model.fit(X_train, y_train)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            # Use higher threshold for more conservative predictions
            y_pred = (y_proba >= 0.6).astype(int)
        else:
            y_pred = model.predict(X_test)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        results[name] = {
            'fp': fp, 'fn': fn, 'tp': tp, 'tn': tn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
    
    print("\n=== Model Comparison for FP Minimization ===")
    results_df = pd.DataFrame(results).T
    print(results_df[['fp', 'fn', 'precision', 'recall']].round(3))
    
    return results

# ============================================================================
# STEP 5: ENSEMBLE METHODS FOR FP REDUCTION
# ============================================================================

def create_fp_optimized_ensemble(X_train, X_test, y_train, y_test):
    """
    Create ensemble specifically designed to minimize FP
    """
    from sklearn.ensemble import VotingClassifier
    
    # Train multiple models with different strategies
    rf1 = RandomForestClassifier(n_estimators=100, class_weight={0: 2.0, 1: 1.0}, random_state=42)
    rf2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=43)  # More conservative
    rf3 = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=44)  # More conservative
    
    # Create ensemble that requires majority agreement
    ensemble = VotingClassifier(
        estimators=[('rf1', rf1), ('rf2', rf2), ('rf3', rf3)],
        voting='hard'  # Requires majority agreement = more conservative
    )
    
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f"\n=== Ensemble for FP Reduction ===")
    print(f"Ensemble results:")
    print(f"  FP: {fp}")
    print(f"  FN: {fn}")
    print(f"  Precision: {tp/(tp+fp):.3f}")
    print(f"  Recall: {tp/(tp+fn):.3f}")
    
    return ensemble

# ============================================================================
# STEP 6: PRODUCTION MONITORING
# ============================================================================

def create_fp_monitoring_system():
    """
    Set up monitoring for FP in production
    """
    print("\n=== Production FP Monitoring ===")
    
    monitoring_config = {
        'fp_threshold': 0.05,  # Alert if FP rate > 5%
        'precision_threshold': 0.8,  # Alert if precision < 80%
        'monitoring_window': '1d',  # Check daily
        'alert_channels': ['email', 'slack']
    }
    
    print("Monitoring Configuration:")
    for key, value in monitoring_config.items():
        print(f"  {key}: {value}")
    
    # Example monitoring function
    def check_fp_metrics(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        alerts = []
        if fp_rate > monitoring_config['fp_threshold']:
            alerts.append(f"FP rate ({fp_rate:.3f}) exceeds threshold")
        if precision < monitoring_config['precision_threshold']:
            alerts.append(f"Precision ({precision:.3f}) below threshold")
        
        return alerts
    
    return check_fp_metrics

# ============================================================================
# STEP 7: IMPLEMENTATION CHECKLIST
# ============================================================================

def fp_minimization_checklist():
    """
    Checklist for implementing FP minimization
    """
    checklist = [
        "□ Analyze business costs of FP vs FN",
        "□ Understand FP patterns in your data",
        "□ Optimize classification threshold",
        "□ Use class weights to penalize FP",
        "□ Consider ensemble methods",
        "□ Implement feature engineering",
        "□ Set up monitoring and alerts",
        "□ Document FP reduction strategy",
        "□ Test with business stakeholders",
        "□ Plan for model retraining"
    ]
    
    print("\n=== FP Minimization Implementation Checklist ===")
    for item in checklist:
        print(item)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Simulate data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.3 > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Convert to DataFrame for easier handling
    X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    
    # Run all steps
    fp_cost, fn_cost = analyze_business_costs()
    
    # Train baseline model
    baseline_model = RandomForestClassifier(random_state=42)
    baseline_model.fit(X_train, y_train)
    
    # Analyze FP patterns
    fp_data = analyze_fp_patterns(X_test_df, y_test, baseline_model)
    
    # Optimize threshold
    y_proba = baseline_model.predict_proba(X_test)[:, 1]
    optimal_threshold, threshold_results = optimize_threshold_for_fp(y_test, y_proba)
    
    # Compare models
    model_results = compare_models_for_fp(X_train_df, X_test_df, y_train, y_test)
    
    # Create ensemble
    ensemble = create_fp_optimized_ensemble(X_train_df, X_test_df, y_train, y_test)
    
    # Set up monitoring
    monitor_func = create_fp_monitoring_system()
    
    # Show checklist
    fp_minimization_checklist()
    
    print("\n=== Implementation Summary ===")
    print("1. Business analysis completed")
    print("2. FP patterns analyzed")
    print("3. Threshold optimized")
    print("4. Models compared")
    print("5. Ensemble created")
    print("6. Monitoring configured")
    print("7. Checklist provided")
    
    print("\nNext steps:")
    print("- Implement the optimal threshold in production")
    print("- Set up the monitoring system")
    print("- Schedule regular model retraining")
    print("- Document the FP reduction strategy") 