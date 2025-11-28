import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings
import joblib

warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Load Data ---
print("Loading data...")
try:
    df_train = pd.read_csv('train_updated.csv')
    df_test = pd.read_csv('test_updated.csv')
except FileNotFoundError as e:
    print("Error: One or both files not found.")
    print(f"Details: {e}")
    exit()

# Store ProfileID for submission
test_profile_ids = df_test['ProfileID']

# Separate features (X) and target (y)
X_train_raw = df_train.drop(['ProfileID', 'RiskFlag'], axis=1)
y_train = df_train['RiskFlag']
X_test_raw = df_test.drop('ProfileID', axis=1)

# Column groups
binary_cols = ['OwnsProperty', 'FamilyObligation', 'JointApplicant']
nominal_cols = ['QualificationLevel', 'WorkCategory', 'RelationshipStatus', 'FundUseCase']
numeric_cols = [
    'ApplicantYears', 'AnnualEarnings', 'RequestedSum', 'TrustMetric',
    'WorkDuration', 'ActiveAccounts', 'OfferRate', 'RepayPeriod', 'DebtFactor'
]

# --- 2. Preprocessing Functions ---
def encode_binary(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            df[col] = df[col].fillna(0).astype(int)
    return df

# --- 3. Apply Preprocessing ---
print("Applying preprocessing...")

X_train_bin = encode_binary(X_train_raw, binary_cols)
X_test_bin = encode_binary(X_test_raw, binary_cols)

# Combine for consistent OHE
X_combined = pd.concat([X_train_bin, X_test_bin], ignore_index=True, sort=False)
X_combined_ohe = pd.get_dummies(X_combined, columns=nominal_cols, drop_first=True)

X_train_ohe = X_combined_ohe.iloc[:len(X_train_bin)].copy()
X_test_ohe = X_combined_ohe.iloc[len(X_train_bin):].copy()

X_train_processed = X_train_ohe.reindex(columns=X_combined_ohe.columns, fill_value=0)
X_test_processed = X_test_ohe.reindex(columns=X_combined_ohe.columns, fill_value=0)

# ================= FEATURE ENGINEERING ================= #

# --- Loan to Income Ratio ---
X_train_processed['LoanToIncome'] = X_train_raw['RequestedSum'] / (X_train_raw['AnnualEarnings'] + 1)
X_test_processed['LoanToIncome'] = X_test_raw['RequestedSum'] / (X_test_raw['AnnualEarnings'] + 1)

# Income per active account
X_train_processed['IncomePerAccount'] = X_train_raw['AnnualEarnings'] / (X_train_raw['ActiveAccounts'] + 1)
X_test_processed['IncomePerAccount'] = X_test_raw['AnnualEarnings'] / (X_test_raw['ActiveAccounts'] + 1)

# EMI burden approximation
X_train_processed['EMI_Burden'] = X_train_raw['RequestedSum'] / (X_train_raw['RepayPeriod'] + 1)
X_test_processed['EMI_Burden'] = X_test_raw['RequestedSum'] / (X_test_raw['RepayPeriod'] + 1)

# Risk intensity
X_train_processed['DebtStress'] = X_train_raw['DebtFactor'] * X_train_raw['RequestedSum']
X_test_processed['DebtStress'] = X_test_raw['DebtFactor'] * X_test_raw['RequestedSum']


# Add engineered features for scaling
numeric_cols.extend([
    'LoanToIncome',
    'IncomePerAccount',
    'EMI_Burden',
    'DebtStress'
])

# ========== GLOBAL SAFETY CLEANUP ==========
X_train_processed = X_train_processed.replace([np.inf, -np.inf], np.nan)
X_test_processed = X_test_processed.replace([np.inf, -np.inf], np.nan)

train_medians = X_train_processed.median()

X_train_processed = X_train_processed.fillna(train_medians)
X_test_processed = X_test_processed.fillna(train_medians)

# --- Handle skewed numeric features ---
skew_cols = ['AnnualEarnings', 'RequestedSum']
for col in skew_cols:
    if col in X_train_processed.columns:
        X_train_processed[col] = np.log1p(X_train_processed[col])
        X_test_processed[col] = np.log1p(X_test_processed[col])

# --- Scaling ---
scaler = StandardScaler()
X_train_processed[numeric_cols] = scaler.fit_transform(X_train_processed[numeric_cols])
X_test_processed[numeric_cols] = scaler.transform(X_test_processed[numeric_cols])

# --- 4. Logistic Regression Model ---

print("Training Logistic Regression model...")

model = LogisticRegression(
    C=0.01,
    penalty='l2',
    solver='lbfgs',
    max_iter=3000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_processed, y_train)


print("Generating predictions...")
predictions = model.predict(X_test_processed)

# Optional: probabilities if needed
probabilities = model.predict_proba(X_test_processed)[:, 1]

# --- 5. Create Submission File ---
submission_df = pd.DataFrame({
    'ProfileID': test_profile_ids,
    'RiskFlag': predictions
})

submission_filename = 'logistic_submission.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\nSubmission file created successfully: {submission_filename}")
print(submission_df.head())

joblib.dump(model, "logistic_model.joblib")

# Save scaler
joblib.dump(scaler, "scaler.joblib")

# Save exact feature order
joblib.dump(list(X_train_processed.columns), "columns.joblib")

print("\nSaved: logistic_model.joblib, scaler.joblib, columns.joblib")
