# inference_test_split_validation.py

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. LOAD MODEL, SCALER, AND TRAIN COLUMNS ================= #
model = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler.joblib')
train_columns = joblib.load('columns.joblib')
print("Model, scaler, and train columns loaded successfully!")

# ================= 2. LOAD TEST DATA ================= #
df_test_full = pd.read_csv("test_updated.csv")

# Check if RiskFlag exists
if 'RiskFlag' not in df_test_full.columns:
    print("Warning: test_updated.csv has no 'RiskFlag'. Validation will require synthetic split for demonstration.")
    # If no RiskFlag, you cannot calculate accuracy; just do preprocessing and predictions
    y_val = None
else:
    # Split 20% for validation
    df_rest, df_val = train_test_split(df_test_full, test_size=0.2, random_state=42, stratify=df_test_full['RiskFlag'])
    X_val_raw = df_val.drop(['ProfileID', 'RiskFlag'], axis=1)
    y_val = df_val['RiskFlag']

# ================= 3. PREPROCESSING FUNCTIONS ================= #
binary_cols = ['OwnsProperty', 'FamilyObligation', 'JointApplicant']
nominal_cols = ['QualificationLevel', 'WorkCategory', 'RelationshipStatus', 'FundUseCase']

def encode_binary(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes':1, 'No':0}).fillna(0).astype(int)
    return df

numeric_cols = [
    'ApplicantYears', 'AnnualEarnings', 'RequestedSum', 'TrustMetric',
    'WorkDuration', 'ActiveAccounts', 'OfferRate', 'RepayPeriod', 'DebtFactor',
    'LoanToIncome', 'IncomePerAccount', 'EMI_Burden', 'DebtStress'
]

# ================= 4. VALIDATION PREPROCESSING ================= #
if y_val is not None:
    X_val_bin = encode_binary(X_val_raw, binary_cols)
    X_val_ohe = pd.get_dummies(X_val_bin, columns=nominal_cols, drop_first=True)
    X_val_processed = X_val_ohe.reindex(columns=train_columns, fill_value=0)

    # Feature engineering
    X_val_processed['LoanToIncome'] = X_val_raw['RequestedSum'] / (X_val_raw['AnnualEarnings'] + 1)
    X_val_processed['IncomePerAccount'] = X_val_raw['AnnualEarnings'] / (X_val_raw['ActiveAccounts'] + 1)
    X_val_processed['EMI_Burden'] = X_val_raw['RequestedSum'] / (X_val_raw['RepayPeriod'] + 1)
    X_val_processed['DebtStress'] = X_val_raw['DebtFactor'] * X_val_raw['RequestedSum']

    X_val_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_val_processed.fillna(X_val_processed.median(), inplace=True)

    # Log transform
    for col in ['AnnualEarnings', 'RequestedSum']:
        if col in X_val_processed.columns:
            X_val_processed[col] = np.log1p(X_val_processed[col])

    # Scale numeric columns
    X_val_processed[numeric_cols] = scaler.transform(X_val_processed[numeric_cols])

    # ================= 5. VALIDATION PREDICTIONS ================= #
    val_preds = model.predict(X_val_processed)
    accuracy = accuracy_score(y_val, val_preds)
    error_pct = 100 * (1 - accuracy)
    print(f"\nValidation Accuracy on 20% split of test_updated: {accuracy:.4f}")
    print(f"Validation Error %: {error_pct:.2f}%")

    # Save predictions
    val_output = pd.DataFrame({
        'ProfileID': df_val['ProfileID'],
        'TrueRiskFlag': y_val,
        'PredictedRiskFlag': val_preds
    })
    val_output.to_csv("validation_predictions.csv", index=False)
    print("Validation predictions saved as validation_predictions.csv")

    # ================= 6. VISUALIZATIONS ================= #
    # Confusion Matrix
    cm = confusion_matrix(y_val, val_preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Validation Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # True vs Predicted distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x=y_val, label='True', color='blue')
    sns.countplot(x=val_preds, label='Predicted', color='orange', alpha=0.5)
    plt.title("True vs Predicted Class Distribution")
    plt.legend(['True','Predicted'])
    plt.show()

# ================= 7. PREDICTIONS ON FULL TEST_UPDATED ================= #
profile_ids_test = df_test_full['ProfileID']
X_test_raw = df_test_full.drop('ProfileID', axis=1)

X_test_bin = encode_binary(X_test_raw, binary_cols)
X_test_ohe = pd.get_dummies(X_test_bin, columns=nominal_cols, drop_first=True)
X_test_processed = X_test_ohe.reindex(columns=train_columns, fill_value=0)

X_test_processed['LoanToIncome'] = X_test_raw['RequestedSum'] / (X_test_raw['AnnualEarnings'] + 1)
X_test_processed['IncomePerAccount'] = X_test_raw['AnnualEarnings'] / (X_test_raw['ActiveAccounts'] + 1)
X_test_processed['EMI_Burden'] = X_test_raw['RequestedSum'] / (X_test_raw['RepayPeriod'] + 1)
X_test_processed['DebtStress'] = X_test_raw['DebtFactor'] * X_test_raw['RequestedSum']

X_test_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_processed.fillna(X_test_processed.median(), inplace=True)

for col in ['AnnualEarnings', 'RequestedSum']:
    if col in X_test_processed.columns:
        X_test_processed[col] = np.log1p(X_test_processed[col])

X_test_processed[numeric_cols] = scaler.transform(X_test_processed[numeric_cols])

# Predict
test_preds = model.predict(X_test_processed)
test_output = pd.DataFrame({
    'ProfileID': profile_ids_test,
    'RiskFlag': test_preds
})
test_output.to_csv("inference_test_updated_predictions.csv", index=False)
print("Full test_updated predictions saved as inference_test_updated_predictions.csv")
# ================== VISUALIZATIONS ==================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1️⃣ Predicted Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=test_preds, palette='pastel')
plt.title("Predicted RiskFlag Distribution (Test Updated)")
plt.xlabel("Predicted RiskFlag")
plt.ylabel("Count")
plt.show()

# 2️⃣ Feature Importance (Linear SVM only)
if hasattr(model, "coef_"):
    coef_abs = np.abs(model.coef_[0])
    feature_names = train_columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': coef_abs})
    top_features = importance_df.sort_values(by='Importance', ascending=False).head(15)
    
    plt.figure(figsize=(8,6))
    sns.barplot(x='Importance', y='Feature', data=top_features, palette="viridis")
    plt.title("Top 15 Feature Importances (Linear SVM)")
    plt.show()

# 3️⃣ Optional: Correlation Heatmap of numeric features
numeric_cols = [
    'ApplicantYears', 'AnnualEarnings', 'RequestedSum', 'TrustMetric',
    'WorkDuration', 'ActiveAccounts', 'OfferRate', 'RepayPeriod', 'DebtFactor',
    'LoanToIncome', 'IncomePerAccount', 'EMI_Burden', 'DebtStress'
]
plt.figure(figsize=(10,8))
sns.heatmap(X_test_processed[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()
