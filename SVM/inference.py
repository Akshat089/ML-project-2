# inference.py

import pandas as pd
import numpy as np
import joblib

# =========================================================
# 1. LOAD MODEL & SCALER
# =========================================================
model = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler.joblib')
print("Model & scaler loaded successfully!")

# =========================================================
# 2. LOAD TEST SUBSET (ONLY ProfileID + ORIGINAL FEATURES)
# =========================================================
df = pd.read_csv("test_subset.csv")     # YOU CREATED THIS
profile_ids = df["ProfileID"]           # KEEP IT
X = df.drop("ProfileID", axis=1)        # REMOVE ONLY ProfileID

# =========================================================
# 3. SAME PREPROCESSING AS TRAINING
# =========================================================

binary_cols = ['OwnsProperty', 'FamilyObligation', 'JointApplicant']
nominal_cols = ['QualificationLevel', 'WorkCategory', 'RelationshipStatus', 'FundUseCase']

def encode_binary(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    return df

X_bin = encode_binary(X, binary_cols)

# DUMMY ENCODING (⚠ IMPORTANT)
X_ohe = pd.get_dummies(X_bin, columns=nominal_cols, drop_first=True)

# LOAD TRAIN COLUMNS
train_columns = joblib.load("columns.joblib")    # we’ll save this in next step

# REINDEX TO SAME COLUMNS AS TRAINING
X_processed = X_ohe.reindex(columns=train_columns, fill_value=0)

# FEATURE ENGINEERING
X_processed['LoanToIncome'] = X['RequestedSum'] / (X['AnnualEarnings'] + 1)
X_processed['IncomePerAccount'] = X['AnnualEarnings'] / (X['ActiveAccounts'] + 1)
X_processed['EMI_Burden'] = X['RequestedSum'] / (X['RepayPeriod'] + 1)
X_processed['DebtStress'] = X['DebtFactor'] * X['RequestedSum']

# Fix columns after creating new ones
if 'LoanToIncome' not in X_processed:
    X_processed['LoanToIncome'] = 0
if 'IncomePerAccount' not in X_processed:
    X_processed['IncomePerAccount'] = 0
if 'EMI_Burden' not in X_processed:
    X_processed['EMI_Burden'] = 0
if 'DebtStress' not in X_processed:
    X_processed['DebtStress'] = 0

# Fill NaN
X_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
X_processed.fillna(X_processed.median(), inplace=True)

# LOG transform
for col in ['AnnualEarnings', 'RequestedSum']:
    if col in X_processed.columns:
        X_processed[col] = np.log1p(X_processed[col])

# SCALE numeric columns
numeric_cols = [
    'ApplicantYears', 'AnnualEarnings', 'RequestedSum', 'TrustMetric',
    'WorkDuration', 'ActiveAccounts', 'OfferRate', 'RepayPeriod', 'DebtFactor',
    'LoanToIncome', 'IncomePerAccount', 'EMI_Burden', 'DebtStress'
]
X_processed[numeric_cols] = scaler.transform(X_processed[numeric_cols])

# =========================================================
# 4. PREDICT
# =========================================================
preds = model.predict(X_processed)
output = pd.DataFrame({"ProfileID": profile_ids, "RiskFlag": preds})

output.to_csv("inference_output.csv", index=False)
print("DONE! Saved: inference_output.csv")
print(output.head())
