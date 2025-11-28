# logistic_inference_testgraphs.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. LOAD MODEL & ARTIFACTS ================= #
print("Loading model, scaler, and training columns...")
model = joblib.load("logistic_model.joblib")
scaler = joblib.load("scaler.joblib")
train_columns = joblib.load("columns.joblib")
print("Artifacts loaded successfully!")

# ================= 2. LOAD TEST DATA ================= #
df_test = pd.read_csv("test_updated.csv")
profile_ids = df_test["ProfileID"]
X_test_raw = df_test.drop("ProfileID", axis=1)

# ================= 3. PREPROCESSING ================= #
binary_cols = ['OwnsProperty', 'FamilyObligation', 'JointApplicant']
nominal_cols = ['QualificationLevel', 'WorkCategory', 'RelationshipStatus', 'FundUseCase']

def encode_binary(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    return df

X_test_bin = encode_binary(X_test_raw, binary_cols)
X_test_ohe = pd.get_dummies(X_test_bin, columns=nominal_cols, drop_first=True)

# Align with training columns
X_test_processed = X_test_ohe.reindex(columns=train_columns, fill_value=0)

# Feature engineering
X_test_processed['LoanToIncome'] = X_test_raw['RequestedSum'] / (X_test_raw['AnnualEarnings'] + 1)
X_test_processed['IncomePerAccount'] = X_test_raw['AnnualEarnings'] / (X_test_raw['ActiveAccounts'] + 1)
X_test_processed['EMI_Burden'] = X_test_raw['RequestedSum'] / (X_test_raw['RepayPeriod'] + 1)
X_test_processed['DebtStress'] = X_test_raw['DebtFactor'] * X_test_raw['RequestedSum']

X_test_processed = X_test_processed.replace([np.inf, -np.inf], np.nan).fillna(0)

# Log transform skewed numeric features
skew_cols = ['AnnualEarnings', 'RequestedSum']
for col in skew_cols:
    if col in X_test_processed.columns:
        X_test_processed[col] = np.log1p(X_test_processed[col])

# Scale numeric features
numeric_cols = ['ApplicantYears', 'AnnualEarnings', 'RequestedSum', 'TrustMetric',
                'WorkDuration', 'ActiveAccounts', 'OfferRate', 'RepayPeriod', 'DebtFactor',
                'LoanToIncome', 'IncomePerAccount', 'EMI_Burden', 'DebtStress']

X_test_processed[numeric_cols] = scaler.transform(X_test_processed[numeric_cols])

# ================= 4. PREDICT ================= #
print("Making predictions on test data...")
predictions = model.predict(X_test_processed)

# ================= 5. SAVE PREDICTIONS ================= #
submission = pd.DataFrame({
    "ProfileID": profile_ids,
    "RiskFlag": predictions
})
submission.to_csv("logistic_test_predictions.csv", index=False)
print("\nSaved: logistic_test_predictions.csv")
print(submission.head())

# ================= 6. PLOTS ================= #
# Distribution of predicted RiskFlag
plt.figure(figsize=(6,4))
sns.countplot(x=predictions)
plt.title("Prediction Distribution on Test Data")
plt.xlabel("Predicted RiskFlag")
plt.ylabel("Count")
plt.show()

# Optional: histogram of LoanToIncome in predictions
plt.figure(figsize=(6,4))
sns.histplot(X_test_processed['LoanToIncome'], bins=20, kde=True)
plt.title("Distribution of LoanToIncome (Test Data)")
plt.xlabel("LoanToIncome")
plt.ylabel("Count")
plt.show()
