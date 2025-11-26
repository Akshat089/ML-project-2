import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np

# --- 1. Load Data ---
print("Loading data...")
try:
    df_train = pd.read_csv('train_updated.csv')
    df_test = pd.read_csv('test_updated.csv')
except FileNotFoundError as e:
    print(f"Error: One or both files not found. Make sure 'train_updated.csv' and 'test_updated.csv' are in the same directory.")
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
# Note: XGBoost is generally less sensitive to scaling, but it's good practice
scaler = StandardScaler()
X_train_processed[numeric_cols] = scaler.fit_transform(X_train_processed[numeric_cols])
X_test_processed[numeric_cols] = scaler.transform(X_test_processed[numeric_cols])

# --- 4. Model Training and Prediction ---

# Calculate Imbalance Weight: Ratio of Negative (0) to Positive (1) class
count_0 = df_train['RiskFlag'].value_counts()[0]
count_1 = df_train['RiskFlag'].value_counts()[1]
scale_pos_weight = count_0 / count_1 # Approximately 7.60

print(f"Training XGBoost model (scale_pos_weight={scale_pos_weight:.2f})...")

# XGBClassifier with sensible initial parameters
model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=500, # Number of boosting rounds
    learning_rate=0.05, # Slower learning often leads to better results
    max_depth=7, # Deeper trees to capture complex interactions
    scale_pos_weight=scale_pos_weight, # Crucial for handling class imbalance
    use_label_encoder=False,
    eval_metric='logloss', # Common metric for binary classification
    random_state=42,
    n_jobs=-1 # Use all available cores
)

model.fit(X_train_processed, y_train)

print("Generating predictions...")
# For a competition using Accuracy/F1-Score, use model.predict().
# For AUC, you might use model.predict_proba()[:, 1] and then choose a threshold.
# We stick to the default model.predict() for a binary submission.
predictions = model.predict(X_test_processed)

# --- 5. Create Submission File ---

submission_df = pd.DataFrame({
    'ProfileID': test_profile_ids,
    'RiskFlag': predictions
})

submission_filename = 'xgboost_submission.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\nSubmission file created successfully: {submission_filename}")
print(submission_df.head())