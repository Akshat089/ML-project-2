import pandas as pd
import joblib
import numpy as np

# --- Load saved artifacts ---
print("Loading model and preprocessing artifacts...")
model = joblib.load("logistic_model.joblib")
scaler = joblib.load("scaler.joblib")
train_columns = joblib.load("columns.joblib")

# --- Load TEST data ---
df_test = pd.read_csv("test_updated.csv")
profile_ids = df_test["ProfileID"]
X_test_raw = df_test.drop("ProfileID", axis=1)

# --- Same preprocessing as training ---
binary_cols = ['OwnsProperty', 'FamilyObligation', 'JointApplicant']
nominal_cols = ['QualificationLevel', 'WorkCategory', 'RelationshipStatus', 'FundUseCase']

def encode_binary(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    return df

X_test_bin = encode_binary(X_test_raw, binary_cols)
X_test_ohe = pd.get_dummies(X_test_bin, columns=nominal_cols, drop_first=True)

# Match EXACT column order as training
X_test_processed = X_test_ohe.reindex(columns=train_columns, fill_value=0)

# Feature engineering (copy-paste from training)
X_test_processed['LoanToIncome'] = X_test_raw['RequestedSum'] / (X_test_raw['AnnualEarnings'] + 1)
X_test_processed['IncomePerAccount'] = X_test_raw['AnnualEarnings'] / (X_test_raw['ActiveAccounts'] + 1)
X_test_processed['EMI_Burden'] = X_test_raw['RequestedSum'] / (X_test_raw['RepayPeriod'] + 1)
X_test_processed['DebtStress'] = X_test_raw['DebtFactor'] * X_test_raw['RequestedSum']

X_test_processed = X_test_processed.replace([np.inf, -np.inf], np.nan).fillna(0)

# Log transform & scale numeric
skew_cols = ['AnnualEarnings', 'RequestedSum']
for col in skew_cols:
    if col in X_test_processed.columns:
        X_test_processed[col] = np.log1p(X_test_processed[col])

# Only scale numeric columns from training file
numeric_cols = ['ApplicantYears', 'AnnualEarnings', 'RequestedSum', 'TrustMetric',
                'WorkDuration', 'ActiveAccounts', 'OfferRate', 'RepayPeriod', 'DebtFactor',
                'LoanToIncome', 'IncomePerAccount', 'EMI_Burden', 'DebtStress']

X_test_processed[numeric_cols] = scaler.transform(X_test_processed[numeric_cols])

# ---> Predict
print("Making predictions...")
predictions = model.predict(X_test_processed)

# Save output
submission = pd.DataFrame({
    "ProfileID": profile_ids,
    "RiskFlag": predictions
})
submission.to_csv("logistic_inference_submission.csv", index=False)
print("\nSaved: logistic_inference_submission.csv")
print(submission.head())
