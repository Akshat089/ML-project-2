# full_pipeline_train_and_save.py

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
import joblib   # <-- ONLY addition

warnings.filterwarnings('ignore', category=UserWarning)

# =========================================================
#                 LOAD DATA
# =========================================================
print("Loading data...")
try:
    df_train = pd.read_csv('train_updated.csv')
    df_test = pd.read_csv('test_updated.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

test_profile_ids = df_test['ProfileID']
X_train_raw = df_train.drop(['ProfileID', 'RiskFlag'], axis=1)
y_train = df_train['RiskFlag']
X_test_raw = df_test.drop('ProfileID', axis=1)

# =========================================================
#                 ADVANCED EDA SECTION (YOUR CODE)
# =========================================================
print("\n--- 🔍 EDA START ---")
print("\n📌 Dataset Info:\n")
print(df_train.info())
print("\n📌 First 5 rows:\n", df_train.head())
print("\n📌 Missing Values (%):\n")
print((df_train.isnull().sum() / len(df_train)) * 100)

print("\n📌 Target Distribution:\n")
print(y_train.value_counts(normalize=True))

numeric_cols = [
    'ApplicantYears', 'AnnualEarnings', 'RequestedSum', 'TrustMetric',
    'WorkDuration', 'ActiveAccounts', 'OfferRate', 'RepayPeriod', 'DebtFactor'
]

print("\n📌 Numerical Statistics:\n")
print(df_train[numeric_cols].describe())

plt.figure(figsize=(12, 8))
sns.heatmap(df_train[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("🔗 Correlation Heatmap")
plt.show()

# for col in numeric_cols:
#     plt.figure(figsize=(6, 3))
#     sns.histplot(df_train[col], kde=True)
#     plt.title(f"📈 Distribution of {col}")
#     plt.show()

# for col in numeric_cols:
#     plt.figure(figsize=(6, 3))
#     sns.boxplot(x=df_train[col])
#     plt.title(f"📦 Boxplot – {col}")
#     plt.show()

# for col in numeric_cols:
#     plt.figure(figsize=(6, 3))
#     sns.boxplot(x=y_train, y=df_train[col])
#     plt.title(f"🎯 {col} vs RiskFlag")
#     plt.show()

categorical_cols = ['QualificationLevel', 'WorkCategory', 'RelationshipStatus',
                    'FundUseCase', 'OwnsProperty', 'FamilyObligation', 'JointApplicant']

for col in categorical_cols:
    if col in df_train.columns:
        plt.figure(figsize=(6, 3))
        sns.countplot(data=df_train, x=col, hue=y_train)
        plt.title(f"📊 {col} by RiskFlag")
        plt.xticks(rotation=45)
        plt.show()

print("\n--- 🔍 EDA END ---")

# =========================================================
# PREPROCESSING + FEATURE ENGINEERING (YOUR CODE)
# =========================================================
binary_cols = ['OwnsProperty', 'FamilyObligation', 'JointApplicant']
nominal_cols = ['QualificationLevel', 'WorkCategory', 'RelationshipStatus', 'FundUseCase']

def encode_binary(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    return df

print("\nApplying preprocessing...")
X_train_bin = encode_binary(X_train_raw, binary_cols)
X_test_bin = encode_binary(X_test_raw, binary_cols)

X_combined = pd.concat([X_train_bin, X_test_bin], ignore_index=True)
X_combined_ohe = pd.get_dummies(X_combined, columns=nominal_cols, drop_first=True)

X_train_ohe = X_combined_ohe.iloc[:len(X_train_bin)].copy()
X_test_ohe = X_combined_ohe.iloc[len(X_train_bin):].copy()

X_train_processed = X_train_ohe.reindex(columns=X_combined_ohe.columns, fill_value=0)
X_test_processed = X_test_ohe.reindex(columns=X_combined_ohe.columns, fill_value=0)

X_train_processed['LoanToIncome'] = X_train_raw['RequestedSum'] / (X_train_raw['AnnualEarnings'] + 1)
X_test_processed['LoanToIncome'] = X_test_raw['RequestedSum'] / (X_test_raw['AnnualEarnings'] + 1)

X_train_processed['IncomePerAccount'] = X_train_raw['AnnualEarnings'] / (X_train_raw['ActiveAccounts'] + 1)
X_test_processed['IncomePerAccount'] = X_test_raw['AnnualEarnings'] / (X_test_raw['ActiveAccounts'] + 1)

X_train_processed['EMI_Burden'] = X_train_raw['RequestedSum'] / (X_train_raw['RepayPeriod'] + 1)
X_test_processed['EMI_Burden'] = X_test_raw['RequestedSum'] / (X_test_raw['RepayPeriod'] + 1)

X_train_processed['DebtStress'] = X_train_raw['DebtFactor'] * X_train_raw['RequestedSum']
X_test_processed['DebtStress'] = X_test_raw['DebtFactor'] * X_test_raw['RequestedSum']

numeric_cols.extend(['LoanToIncome', 'IncomePerAccount', 'EMI_Burden', 'DebtStress'])
X_train_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_processed.replace([np.inf, -np.inf], np.nan, inplace=True)

train_medians = X_train_processed.median()
X_train_processed.fillna(train_medians, inplace=True)
X_test_processed.fillna(train_medians, inplace=True)

skew_cols = ['AnnualEarnings', 'RequestedSum']
for col in skew_cols:
    X_train_processed[col] = np.log1p(X_train_processed[col])
    X_test_processed[col] = np.log1p(X_test_processed[col])

scaler = StandardScaler()
X_train_processed[numeric_cols] = scaler.fit_transform(X_train_processed[numeric_cols])
X_test_processed[numeric_cols] = scaler.transform(X_test_processed[numeric_cols])

print("\nTraining LinearSVC model...")
model = LinearSVC(random_state=42, max_iter=5000, dual=False, C=0.1)
model.fit(X_train_processed, y_train)

print("\nGenerating predictions...")
predictions = model.predict(X_test_processed)

submission_df = pd.DataFrame({'ProfileID': test_profile_ids, 'RiskFlag': predictions})
submission_df.to_csv('svm_FE_submission.csv', index=False)
print("\nSubmission file created successfully: svm_FE_submission.csv")
print(submission_df.head())

# =========================================================
# SAVE MODEL & SCALER
# =========================================================
joblib.dump(list(X_train_processed.columns), "columns.joblib")   # <-- NEW LINE
joblib.dump(model, 'svm_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("\nSaved: svm_model.joblib and scaler.joblib")
