import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np
import warnings
# Suppress the UserWarning about large number of features for LinearSVC
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Load Data ---
print("Loading data...")
try:
    df_train = pd.read_csv('train_updated.csv')
    df_test = pd.read_csv('test_updated.csv')
except FileNotFoundError as e:
    print(f"Error: One or both files not found. Make sure 'train_updated.csv' and 'test_updated.csv' are in the same directory.")
    print(f"Details: {e}")
    exit()

# Store ProfileID for submission
test_profile_ids = df_test['ProfileID']

# Separate features (X) and target (y)
X_train_raw = df_train.drop(['ProfileID', 'RiskFlag'], axis=1)
y_train = df_train['RiskFlag']
X_test_raw = df_test.drop('ProfileID', axis=1)

# List of columns by type
binary_cols = ['OwnsProperty', 'FamilyObligation', 'JointApplicant']
nominal_cols = ['QualificationLevel', 'WorkCategory', 'RelationshipStatus', 'FundUseCase']
numeric_cols = [
    'ApplicantYears', 'AnnualEarnings', 'RequestedSum', 'TrustMetric',
    'WorkDuration', 'ActiveAccounts', 'OfferRate', 'RepayPeriod', 'DebtFactor'
]

# --- 2. Preprocessing Functions ---

def encode_binary(df, columns):
    """Converts 'Yes'/'No' to 1/0 for specified columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            # Handle potential NaN values if they existed, though data inspection suggested none
            df[col] = df[col].replace({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    return df

# --- 3. Apply Preprocessing ---

print("Applying preprocessing...")

# Apply binary encoding
X_train_bin = encode_binary(X_train_raw, binary_cols)
X_test_bin = encode_binary(X_test_raw, binary_cols)

# Combine for consistent One-Hot Encoding
X_combined = pd.concat([X_train_bin, X_test_bin], ignore_index=True, sort=False)
X_combined_ohe = pd.get_dummies(X_combined, columns=nominal_cols, drop_first=True)

# Split back and ensure same columns are present in both
X_train_ohe = X_combined_ohe.iloc[:len(X_train_bin)].copy()
X_test_ohe = X_combined_ohe.iloc[len(X_train_bin):].copy()

# Ensure feature consistency
X_train_processed = X_train_ohe.reindex(columns=X_combined_ohe.columns, fill_value=0)
X_test_processed = X_test_ohe.reindex(columns=X_combined_ohe.columns, fill_value=0)

# Apply scaling to numerical columns
scaler = StandardScaler()
X_train_processed[numeric_cols] = scaler.fit_transform(X_train_processed[numeric_cols])
X_test_processed[numeric_cols] = scaler.transform(X_test_processed[numeric_cols])

# --- 4. Model Training and Prediction ---

print("Training LinearSVC model...")
# LinearSVC is used for efficiency on large datasets.
# dual=False is set because n_samples (200k+) > n_features (~40 after OHE)
model = LinearSVC(random_state=42, max_iter=5000, dual=False, C=0.1)
model.fit(X_train_processed, y_train)

print("Generating predictions...")
predictions = model.predict(X_test_processed)

# --- 5. Create Submission File ---

submission_df = pd.DataFrame({
    'ProfileID': test_profile_ids,
    'RiskFlag': predictions
})

submission_filename = 'svm_submission_from_code.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\nSubmission file created successfully: {submission_filename}")
print(submission_df.head())