import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Simulate data
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 5)
y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.3 > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print("=== AUC and FP Minimization Connection ===")
print(f"Overall AUC: {roc_auc:.3f}")

# ============================================================================
# 1. AUC REPRESENTS MODEL'S ABILITY TO SEPARATE CLASSES
# ============================================================================

def analyze_auc_interpretation():
    """
    AUC tells us how well the model can distinguish between classes
    Higher AUC = better separation = more flexibility in threshold tuning
    """
    print("\n=== 1. AUC Interpretation ===")
    print(f"AUC = {roc_auc:.3f}")
    
    if roc_auc > 0.9:
        print("Excellent separation - you have great flexibility in threshold tuning")
    elif roc_auc > 0.8:
        print("Good separation - reasonable flexibility in threshold tuning")
    elif roc_auc > 0.7:
        print("Fair separation - limited flexibility in threshold tuning")
    else:
        print("Poor separation - threshold tuning may not help much")
    
    print("\nAUC = 0.5 means random guessing")
    print("AUC = 1.0 means perfect separation")
    print("Higher AUC = more room to optimize FP vs FN trade-off")

# ============================================================================
# 2. AUC AND THRESHOLD OPTIMIZATION
# ============================================================================

def analyze_threshold_impact_on_auc():
    """
    Show how different thresholds affect FP/FN while AUC remains constant
    """
    print("\n=== 2. Threshold Selection and AUC ===")
    
    # Test different thresholds
    test_thresholds = [0.3, 0.5, 0.7, 0.9]
    results = []
    
    for threshold in test_thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calculate metrics at this threshold
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'fp': fp, 'fn': fn, 'tp': tp, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'fp_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
        })
    
    results_df = pd.DataFrame(results)
    print("Impact of threshold selection (AUC remains constant):")
    print(results_df[['threshold', 'fp', 'fn', 'precision', 'recall', 'fp_rate']].round(3))
    
    return results_df

# ============================================================================
# 3. AUC AND MODEL COMPARISON
# ============================================================================

def compare_models_auc_fp():
    """
    Compare different models using AUC and their FP minimization potential
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    print("\n=== 3. Model Comparison: AUC vs FP Potential ===")
    
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    comparison_results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        y_proba_model = model.predict_proba(X_test)[:, 1]
        
        # Calculate AUC
        fpr_model, tpr_model, _ = roc_curve(y_test, y_proba_model)
        auc_model = auc(fpr_model, tpr_model)
        
        # Calculate FP at different thresholds
        thresholds_test = [0.3, 0.5, 0.7]
        fp_results = []
        
        for thresh in thresholds_test:
            y_pred_thresh = (y_proba_model >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
            fp_results.append(fp)
        
        comparison_results[name] = {
            'auc': auc_model,
            'fp_at_0.3': fp_results[0],
            'fp_at_0.5': fp_results[1],
            'fp_at_0.7': fp_results[2],
            'fp_reduction_potential': fp_results[0] - fp_results[2]  # How much FP can be reduced
        }
    
    comparison_df = pd.DataFrame(comparison_results).T
    print("Model comparison (AUC vs FP minimization potential):")
    print(comparison_df.round(3))
    
    return comparison_df

# ============================================================================
# 4. AUC AND FEATURE ENGINEERING
# ============================================================================

def analyze_feature_engineering_auc_impact():
    """
    Show how feature engineering affects AUC and FP minimization
    """
    print("\n=== 4. Feature Engineering Impact on AUC ===")
    
    # Baseline features
    X_baseline = X_train[:, :3]  # Use only first 3 features
    X_test_baseline = X_test[:, :3]
    
    # Engineered features (add polynomial features)
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_engineered = poly.fit_transform(X_baseline)
    X_test_engineered = poly.transform(X_test_baseline)
    
    # Compare models
    baseline_model = RandomForestClassifier(random_state=42)
    engineered_model = RandomForestClassifier(random_state=42)
    
    baseline_model.fit(X_baseline, y_train)
    engineered_model.fit(X_engineered, y_train)
    
    # Calculate AUC for both
    y_proba_baseline = baseline_model.predict_proba(X_test_baseline)[:, 1]
    y_proba_engineered = engineered_model.predict_proba(X_test_engineered)[:, 1]
    
    fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_proba_baseline)
    fpr_engineered, tpr_engineered, _ = roc_curve(y_test, y_proba_engineered)
    
    auc_baseline = auc(fpr_baseline, tpr_baseline)
    auc_engineered = auc(fpr_engineered, tpr_engineered)
    
    print(f"Baseline features AUC: {auc_baseline:.3f}")
    print(f"Engineered features AUC: {auc_engineered:.3f}")
    print(f"AUC improvement: {auc_engineered - auc_baseline:.3f}")
    
    # Show FP reduction potential
    y_pred_baseline = (y_proba_baseline >= 0.7).astype(int)
    y_pred_engineered = (y_proba_engineered >= 0.7).astype(int)
    
    _, fp_baseline, _, _ = confusion_matrix(y_test, y_pred_baseline).ravel()
    _, fp_engineered, _, _ = confusion_matrix(y_test, y_pred_engineered).ravel()
    
    print(f"FP at threshold 0.7 (baseline): {fp_baseline}")
    print(f"FP at threshold 0.7 (engineered): {fp_engineered}")
    print(f"FP reduction: {fp_baseline - fp_engineered}")
    
    return {
        'baseline_auc': auc_baseline,
        'engineered_auc': auc_engineered,
        'baseline_fp': fp_baseline,
        'engineered_fp': fp_engineered
    }

# ============================================================================
# 5. AUC AND CLASS WEIGHTS
# ============================================================================

def analyze_class_weights_auc_impact():
    """
    Show how class weights affect AUC and FP minimization
    """
    print("\n=== 5. Class Weights Impact on AUC ===")
    
    # Train models with different class weights
    weights_configs = {
        'Balanced': {0: 1.0, 1: 1.0},
        'FP_Penalized': {0: 2.0, 1: 1.0},  # Penalize FP more
        'FN_Penalized': {0: 1.0, 2: 1.0}   # Penalize FN more
    }
    
    weight_results = {}
    
    for name, weights in weights_configs.items():
        model = RandomForestClassifier(class_weight=weights, random_state=42)
        model.fit(X_train, y_train)
        y_proba_weighted = model.predict_proba(X_test)[:, 1]
        
        # Calculate AUC
        fpr_weighted, tpr_weighted, _ = roc_curve(y_test, y_proba_weighted)
        auc_weighted = auc(fpr_weighted, tpr_weighted)
        
        # Calculate FP at different thresholds
        thresholds = [0.3, 0.5, 0.7]
        fp_values = []
        
        for thresh in thresholds:
            y_pred_thresh = (y_proba_weighted >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
            fp_values.append(fp)
        
        weight_results[name] = {
            'auc': auc_weighted,
            'fp_at_0.3': fp_values[0],
            'fp_at_0.5': fp_values[1],
            'fp_at_0.7': fp_values[2]
        }
    
    weight_df = pd.DataFrame(weight_results).T
    print("Class weights impact on AUC and FP:")
    print(weight_df.round(3))
    
    return weight_df

# ============================================================================
# 6. VISUALIZATION: AUC AND FP MINIMIZATION
# ============================================================================

def visualize_auc_fp_connection():
    """
    Create visualizations showing the connection between AUC and FP minimization
    """
    print("\n=== 6. Visualizing AUC and FP Connection ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: ROC Curve with different threshold points
    axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    # Mark different threshold points
    threshold_points = [0.3, 0.5, 0.7]
    colors = ['red', 'green', 'blue']
    
    for i, threshold in enumerate(threshold_points):
        y_pred_thresh = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
        fpr_point = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_point = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        axes[0, 0].scatter(fpr_point, tpr_point, color=colors[i], s=100, 
                           label=f'Threshold {threshold}')
    
    axes[0, 0].set_xlabel('False Positive Rate (FPR)')
    axes[0, 0].set_ylabel('True Positive Rate (TPR)')
    axes[0, 0].set_title('ROC Curve with Threshold Points')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: FP vs Threshold for different AUC levels
    # Simulate models with different AUCs
    auc_levels = [0.6, 0.7, 0.8, 0.9]
    thresholds_plot = np.arange(0.1, 0.9, 0.05)
    
    for auc_level in auc_levels:
        # Simulate different AUC by adjusting probabilities
        if auc_level < roc_auc:
            # Degrade AUC
            y_proba_adjusted = y_proba * auc_level / roc_auc
        else:
            # Improve AUC
            y_proba_adjusted = y_proba + (auc_level - roc_auc) * 0.5
        
        fp_values = []
        for thresh in thresholds_plot:
            y_pred_thresh = (y_proba_adjusted >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
            fp_values.append(fp)
        
        axes[0, 1].plot(thresholds_plot, fp_values, label=f'AUC = {auc_level}')
    
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('False Positives (FP)')
    axes[0, 1].set_title('FP vs Threshold for Different AUC Levels')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    axes[1, 0].plot(recall, precision, 'g-', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: AUC vs FP minimization potential
    # Show how higher AUC enables better FP minimization
    auc_values = np.arange(0.5, 1.0, 0.05)
    fp_reduction_potential = []
    
    for auc_val in auc_values:
        # Simulate FP reduction potential based on AUC
        # Higher AUC = more room to optimize threshold
        potential = (auc_val - 0.5) * 100  # Scale for visualization
        fp_reduction_potential.append(potential)
    
    axes[1, 1].plot(auc_values, fp_reduction_potential, 'purple', linewidth=2)
    axes[1, 1].set_xlabel('AUC')
    axes[1, 1].set_ylabel('FP Reduction Potential')
    axes[1, 1].set_title('AUC vs FP Minimization Potential')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 7. PRACTICAL GUIDANCE
# ============================================================================

def provide_auc_fp_guidance():
    """
    Provide practical guidance on using AUC for FP minimization
    """
    print("\n=== 7. Practical Guidance: AUC for FP Minimization ===")
    
    print("\nKey Insights:")
    print("1. AUC measures model's ability to separate classes")
    print("2. Higher AUC = more flexibility in threshold tuning")
    print("3. AUC is threshold-independent - it's a model quality metric")
    print("4. Use AUC to compare models, then tune threshold for FP")
    
    print("\nWorkflow:")
    print("1. Compare models using AUC")
    print("2. Select model with highest AUC")
    print("3. Use threshold tuning to minimize FP")
    print("4. Monitor both AUC and FP in production")
    
    print("\nAUC Interpretation for FP Minimization:")
    if roc_auc > 0.8:
        print(f"✓ Good AUC ({roc_auc:.3f}) - you have flexibility to minimize FP")
        print("  - Can use higher thresholds")
        print("  - Feature engineering likely to help")
        print("  - Class weights effective")
    elif roc_auc > 0.7:
        print(f"⚠ Fair AUC ({roc_auc:.3f}) - moderate flexibility for FP minimization")
        print("  - Focus on feature engineering")
        print("  - Consider ensemble methods")
        print("  - Monitor threshold carefully")
    else:
        print(f"✗ Low AUC ({roc_auc:.3f}) - limited ability to minimize FP")
        print("  - Improve model quality first")
        print("  - Focus on feature engineering")
        print("  - Consider different algorithms")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run all analyses
    analyze_auc_interpretation()
    threshold_results = analyze_threshold_impact_on_auc()
    model_comparison = compare_models_auc_fp()
    feature_impact = analyze_feature_engineering_auc_impact()
    weight_impact = analyze_class_weights_auc_impact()
    visualize_auc_fp_connection()
    provide_auc_fp_guidance()
    
    print("\n=== Summary: AUC and FP Minimization ===")
    print("✓ AUC measures model quality and separation ability")
    print("✓ Higher AUC = more flexibility in threshold tuning")
    print("✓ Use AUC to compare models, then tune threshold for FP")
    print("✓ Monitor both AUC and FP in production")
    print("✓ Feature engineering can improve both AUC and FP minimization") 