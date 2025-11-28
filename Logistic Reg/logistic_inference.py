# logistic_inference_validation.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# ================= 1. LOAD ARTIFACTS ================= #
print("Loading model, scaler, and training columns...")
model = joblib.load("logistic_model.joblib")
scaler = joblib.load("scaler.joblib")
train_columns = joblib.load("columns.joblib")
print("Artifacts loaded successfully!")

# ================= 2. LOAD TRAIN DATA ================= #
df_train = pd.read_csv("train_updated.csv")

# Split 20% for subset, 80% for validation
subset_fraction = 0.2
df_subset = df_train.sample(frac=subset_fraction, random_state=42)
df_rest = df_train.drop(df_subset.index)

# Separate features & labels
X_subset = df_subset.drop(["ProfileID", "RiskFlag"], axis=1)
y_subset = df_subset["RiskFlag"]

X_rest = df_rest.drop(["ProfileID", "RiskFlag"], axis=1)
y_rest = df_rest["RiskFlag"]  # True labels for validation

profile_ids_subset = df_subset["ProfileID"]

# ================= 3. PREPROCESSING FUNCTIONS ================= #
binary_cols = ['OwnsProperty', 'FamilyObligation', 'JointApplicant']
nominal_cols = ['QualificationLevel', 'WorkCategory', 'RelationshipStatus', 'FundUseCase']

def encode_binary(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    return df

def preprocess(df_raw):
    df_bin = encode_binary(df_raw, binary_cols)
    df_ohe = pd.get_dummies(df_bin, columns=nominal_cols, drop_first=True)
    df_processed = df_ohe.reindex(columns=train_columns, fill_value=0)

    # Feature engineering
    df_processed['LoanToIncome'] = df_raw['RequestedSum'] / (df_raw['AnnualEarnings'] + 1)
    df_processed['IncomePerAccount'] = df_raw['AnnualEarnings'] / (df_raw['ActiveAccounts'] + 1)
    df_processed['EMI_Burden'] = df_raw['RequestedSum'] / (df_raw['RepayPeriod'] + 1)
    df_processed['DebtStress'] = df_raw['DebtFactor'] * df_raw['RequestedSum']

    df_processed = df_processed.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Log transform skewed numeric columns
    for col in ['AnnualEarnings', 'RequestedSum']:
        if col in df_processed.columns:
            df_processed[col] = np.log1p(df_processed[col])

    # Scale numeric columns
    numeric_cols = ['ApplicantYears', 'AnnualEarnings', 'RequestedSum', 'TrustMetric',
                    'WorkDuration', 'ActiveAccounts', 'OfferRate', 'RepayPeriod', 'DebtFactor',
                    'LoanToIncome', 'IncomePerAccount', 'EMI_Burden', 'DebtStress']
    df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])

    return df_processed

# ================= 4. PROCESS SUBSET & REST ================= #
X_subset_processed = preprocess(X_subset)
X_rest_processed = preprocess(X_rest)

# ================= 5. PREDICTIONS ================= #
print("Making predictions on subset and validation set...")
subset_preds = model.predict(X_subset_processed)
rest_preds = model.predict(X_rest_processed)

# ================= 6. VALIDATION METRICS ================= #
accuracy = accuracy_score(y_rest, rest_preds)
error_percent = 100 * (1 - accuracy)
print(f"\nValidation Accuracy (80% rest): {accuracy:.4f}")
print(f"Validation Error %: {error_percent:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_rest, rest_preds)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix on Validation Set")
plt.show()

# Distribution of predictions
plt.figure(figsize=(6,4))
sns.countplot(x=subset_preds)
plt.title("Prediction Distribution on 20% Subset")
plt.xlabel("Predicted RiskFlag")
plt.ylabel("Count")
plt.show()

# ================= 6b. PRINT DETAILED METRICS ================= #
print("\nClassification Report on Validation Set (80% rest):")
report = classification_report(y_rest, rest_preds, target_names=['Class 0', 'Class 1'])
print(report)

# ================= 7. SAVE SUBSET PREDICTIONS ================= #
submission = pd.DataFrame({
    "ProfileID": profile_ids_subset,
    "RiskFlag": subset_preds
})
submission.to_csv("logistic_subset_predictions.csv", index=False)
print("\nSaved predictions for subset: logistic_subset_predictions.csv")
print(submission.head())
