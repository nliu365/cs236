import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Simulate credit risk data with more realistic features
np.random.seed(42)
n_samples = 2000

# Create realistic credit risk features
data = {
    'income': np.random.lognormal(10, 0.5, n_samples),
    'credit_score': np.random.normal(700, 100, n_samples),
    'debt_ratio': np.random.beta(2, 5, n_samples),
    'payment_history': np.random.normal(0.8, 0.2, n_samples),
    'age': np.random.normal(45, 15, n_samples),
    'employment_length': np.random.exponential(5, n_samples),
    'loan_amount': np.random.lognormal(10, 0.8, n_samples),
    'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
    'home_ownership': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'education': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1])
}

df = pd.DataFrame(data)

# Create target: risky (1) vs not risky (0)
# Higher risk for: low income, low credit score, high debt ratio, poor payment history
risk_score = (
    -0.3 * (df['income'] - df['income'].mean()) / df['income'].std() +
    -0.4 * (df['credit_score'] - df['credit_score'].mean()) / df['credit_score'].std() +
    0.5 * (df['debt_ratio'] - df['debt_ratio'].mean()) / df['debt_ratio'].std() +
    0.3 * (df['payment_history'] - df['payment_history'].mean()) / df['payment_history'].std() +
    np.random.normal(0, 0.3, n_samples)
)

df['risky'] = (risk_score > 0).astype(int)

print("=== Strategy 1: Feature Selection for FP Reduction ===")

# Split data
X = df.drop('risky', axis=1)
y = df['risky']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def evaluate_fp(model, X_test, y_test, threshold=0.5):
    """Evaluate model focusing on FP"""
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
        'fp_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
    }

# Method 1: Select features that are most predictive of FP reduction
print("1.1: Statistical Feature Selection")
selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
model_selected.fit(X_train_selected, y_train)
results_selected = evaluate_fp(model_selected, X_test_selected, y_test)

print(f"With feature selection:")
print(f"  FP: {results_selected['fp']}")
print(f"  Precision: {results_selected['precision']:.3f}")

# Method 2: Recursive Feature Elimination
print("\n1.2: Recursive Feature Elimination")
estimator = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(estimator, n_features_to_select=5)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

rfe_features = X.columns[rfe.support_]
print(f"RFE selected features: {list(rfe_features)}")

model_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
model_rfe.fit(X_train_rfe, y_train)
results_rfe = evaluate_fp(model_rfe, X_test_rfe, y_test)

print(f"With RFE:")
print(f"  FP: {results_rfe['fp']}")
print(f"  Precision: {results_rfe['precision']:.3f}")

print("\n=== Strategy 2: Domain-Specific Feature Engineering ===")

# Create domain-specific features that help reduce FP
df_engineered = df.copy()

# 2.1: Risk ratios and combinations
df_engineered['income_to_debt'] = df_engineered['income'] / (df_engineered['debt_ratio'] + 0.01)
df_engineered['credit_to_debt'] = df_engineered['credit_score'] / (df_engineered['debt_ratio'] * 1000 + 0.01)
df_engineered['payment_to_debt'] = df_engineered['payment_history'] / (df_engineered['debt_ratio'] + 0.01)

# 2.2: Age and experience features
df_engineered['age_employment_ratio'] = df_engineered['age'] / (df_engineered['employment_length'] + 0.01)
df_engineered['experience_level'] = np.where(df_engineered['employment_length'] > 5, 1, 0)

# 2.3: Loan-specific features
df_engineered['loan_to_income'] = df_engineered['loan_amount'] / (df_engineered['income'] + 0.01)
df_engineered['monthly_payment'] = df_engineered['loan_amount'] / df_engineered['loan_term']

# 2.4: Risk buckets
df_engineered['credit_risk_bucket'] = pd.cut(df_engineered['credit_score'], 
                                            bins=[0, 600, 650, 700, 750, 850], 
                                            labels=[0, 1, 2, 3, 4])
df_engineered['income_risk_bucket'] = pd.cut(df_engineered['income'], 
                                             bins=[0, 30000, 50000, 75000, 100000, np.inf], 
                                             labels=[0, 1, 2, 3, 4])

# Convert categorical to numeric
df_engineered['credit_risk_bucket'] = df_engineered['credit_risk_bucket'].astype(int)
df_engineered['income_risk_bucket'] = df_engineered['income_risk_bucket'].astype(int)

# Split engineered data
X_engineered = df_engineered.drop('risky', axis=1)
y_engineered = df_engineered['risky']
X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
    X_engineered, y_engineered, test_size=0.3, random_state=42
)

model_engineered = RandomForestClassifier(n_estimators=100, random_state=42)
model_engineered.fit(X_train_eng, y_train_eng)
results_engineered = evaluate_fp(model_engineered, X_test_eng, y_test_eng)

print(f"With domain-specific features:")
print(f"  FP: {results_engineered['fp']}")
print(f"  Precision: {results_engineered['precision']:.3f}")

print("\n=== Strategy 3: Feature Interactions ===")

# 3.1: Polynomial features for key risk indicators
key_features = ['credit_score', 'debt_ratio', 'payment_history', 'income']
X_key = X[key_features]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_key)
feature_names = poly.get_feature_names_out(key_features)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly, y, test_size=0.3, random_state=42
)

model_poly = RandomForestClassifier(n_estimators=100, random_state=42)
model_poly.fit(X_train_poly, y_train_poly)
results_poly = evaluate_fp(model_poly, X_test_poly, y_test_poly)

print(f"With polynomial features:")
print(f"  FP: {results_poly['fp']}")
print(f"  Precision: {results_poly['precision']:.3f}")

# 3.2: Custom interaction features
df_interactions = df.copy()
df_interactions['credit_debt_interaction'] = df_interactions['credit_score'] * df_interactions['debt_ratio']
df_interactions['income_payment_interaction'] = df_interactions['income'] * df_interactions['payment_history']
df_interactions['age_credit_interaction'] = df_interactions['age'] * df_interactions['credit_score'] / 1000

X_interactions = df_interactions.drop('risky', axis=1)
y_interactions = df_interactions['risky']
X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(
    X_interactions, y_interactions, test_size=0.3, random_state=42
)

model_interactions = RandomForestClassifier(n_estimators=100, random_state=42)
model_interactions.fit(X_train_int, y_train_int)
results_interactions = evaluate_fp(model_interactions, X_test_int, y_test_int)

print(f"With custom interactions:")
print(f"  FP: {results_interactions['fp']}")
print(f"  Precision: {results_interactions['precision']:.3f}")

print("\n=== Strategy 4: Feature Scaling and Normalization ===")

# 4.1: Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = RandomForestClassifier(n_estimators=100, random_state=42)
model_scaled.fit(X_train_scaled, y_train)
results_scaled = evaluate_fp(model_scaled, X_test_scaled, y_test)

print(f"With standard scaling:")
print(f"  FP: {results_scaled['fp']}")
print(f"  Precision: {results_scaled['precision']:.3f}")

# 4.2: Robust scaling (less sensitive to outliers)
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train)
X_test_robust = robust_scaler.transform(X_test)

model_robust = RandomForestClassifier(n_estimators=100, random_state=42)
model_robust.fit(X_train_robust, y_train)
results_robust = evaluate_fp(model_robust, X_test_robust, y_test)

print(f"With robust scaling:")
print(f"  FP: {results_robust['fp']}")
print(f"  Precision: {results_robust['precision']:.3f}")

print("\n=== Strategy 5: Feature Importance Analysis ===")

# Train model and analyze feature importance
model_importance = RandomForestClassifier(n_estimators=100, random_state=42)
model_importance.fit(X_train, y_train)

# Get feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model_importance.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 5 most important features:")
print(importance_df.head())

# Select only top features
top_features = importance_df.head(5)['feature'].values
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

model_top = RandomForestClassifier(n_estimators=100, random_state=42)
model_top.fit(X_train_top, y_train)
results_top = evaluate_fp(model_top, X_test_top, y_test)

print(f"\nWith top 5 features only:")
print(f"  FP: {results_top['fp']}")
print(f"  Precision: {results_top['precision']:.3f}")

# Summary comparison
print("\n=== Summary: Feature Engineering Impact on FP ===")
methods = {
    'Baseline': evaluate_fp(RandomForestClassifier(random_state=42).fit(X_train, y_train), X_test, y_test),
    'Statistical Selection': results_selected,
    'RFE': results_rfe,
    'Domain Features': results_engineered,
    'Polynomial': results_poly,
    'Custom Interactions': results_interactions,
    'Standard Scaling': results_scaled,
    'Robust Scaling': results_robust,
    'Top Features': results_top
}

summary_df = pd.DataFrame(methods).T
print(summary_df[['fp', 'precision', 'recall']].round(3))

# Plot FP comparison
plt.figure(figsize=(12, 6))
methods_list = list(methods.keys())
fp_values = [methods[m]['fp'] for m in methods_list]

plt.bar(methods_list, fp_values, color='skyblue')
plt.title('False Positives (FP) Comparison Across Feature Engineering Methods')
plt.xlabel('Method')
plt.ylabel('False Positives')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show() 