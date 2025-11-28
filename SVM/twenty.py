import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=UserWarning)

# =========================================================
#                     LOAD DATA
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
#                     ADVANCED EDA
# =========================================================
print("\n--- 🔍 EDA START ---")
print("\nDataset Info:\n")
print(df_train.info())

print("\nFirst 5 rows:\n", df_train.head())
print("\nMissing Values (%):\n", (df_train.isnull().sum() / len(df_train)) * 100)
print("\nTarget Distribution:\n", y_train.value_counts(normalize=True))

numeric_cols = [
    'ApplicantYears', 'AnnualEarnings', 'RequestedSum', 'TrustMetric',
    'WorkDuration', 'ActiveAccounts', 'OfferRate', 'RepayPeriod', 'DebtFactor'
]

print("\nNumerical Statistics:\n", df_train[numeric_cols].describe())

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_train[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Distribution and boxplots
# for col in numeric_cols:
#     plt.figure(figsize=(6, 3))
#     sns.histplot(df_train[col], kde=True)
#     plt.title(f"Distribution of {col}")
#     plt.show()
    
#     plt.figure(figsize=(6, 3))
#     sns.boxplot(x=df_train[col])
#     plt.title(f"Boxplot of {col}")
#     plt.show()

# # Categorical counts
# categorical_cols = ['QualificationLevel', 'WorkCategory', 'RelationshipStatus', 'FundUseCase',
#                     'OwnsProperty', 'FamilyObligation', 'JointApplicant']
# for col in categorical_cols:
#     plt.figure(figsize=(6, 3))
#     sns.countplot(data=df_train, x=col, hue=y_train)
#     plt.title(f"{col} by RiskFlag")
#     plt.xticks(rotation=45)
#     plt.show()

print("\n--- 🔍 EDA END ---")

# =========================================================
#               PREPROCESSING + FEATURE ENGINEERING
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

# Feature engineering
X_train_processed['LoanToIncome'] = X_train_raw['RequestedSum'] / (X_train_raw['AnnualEarnings'] + 1)
X_test_processed['LoanToIncome'] = X_test_raw['RequestedSum'] / (X_test_raw['AnnualEarnings'] + 1)

X_train_processed['IncomePerAccount'] = X_train_raw['AnnualEarnings'] / (X_train_raw['ActiveAccounts'] + 1)
X_test_processed['IncomePerAccount'] = X_test_raw['AnnualEarnings'] / (X_test_raw['ActiveAccounts'] + 1)

X_train_processed['EMI_Burden'] = X_train_raw['RequestedSum'] / (X_train_raw['RepayPeriod'] + 1)
X_test_processed['EMI_Burden'] = X_test_raw['RequestedSum'] / (X_test_raw['RepayPeriod'] + 1)

X_train_processed['DebtStress'] = X_train_raw['DebtFactor'] * X_train_raw['RequestedSum']
X_test_processed['DebtStress'] = X_test_raw['DebtFactor'] * X_test_raw['RequestedSum']

numeric_cols.extend(['LoanToIncome', 'IncomePerAccount', 'EMI_Burden', 'DebtStress'])

# Replace inf / NaN
X_train_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
train_medians = X_train_processed.median()
X_train_processed.fillna(train_medians, inplace=True)
X_test_processed.fillna(train_medians, inplace=True)

# Log transform skewed features
skew_cols = ['AnnualEarnings', 'RequestedSum']
for col in skew_cols:
    X_train_processed[col] = np.log1p(X_train_processed[col])
    X_test_processed[col] = np.log1p(X_test_processed[col])

# Scaling
scaler = StandardScaler()
X_train_processed[numeric_cols] = scaler.fit_transform(X_train_processed[numeric_cols])
X_test_processed[numeric_cols] = scaler.transform(X_test_processed[numeric_cols])

# =========================================================
#            TRAIN NORMAL SVM ON 20% DATA
# =========================================================
print("\nTraining SVM on 20% of data...")
X_train_20, _, y_train_20, _ = train_test_split(
    X_train_processed, y_train, test_size=0.8, random_state=42, stratify=y_train
)

svm_20 = SVC(kernel='rbf', C=1, gamma='scale')
svm_20.fit(X_train_20, y_train_20)
pred_20 = svm_20.predict(X_test_processed)

print("Prediction done on 20% data subset.")

# =========================================================
#            TRAIN NORMAL SVM ON 100% DATA
# =========================================================
print("\nTraining SVM on FULL data...")
svm_full = SVC(kernel='rbf', C=1, gamma='scale')
svm_full.fit(X_train_processed, y_train)
pred_full = svm_full.predict(X_test_processed)

print("Prediction done on FULL dataset.")

# =========================================================
#                   SAVE SUBMISSION
# =========================================================
submission_20 = pd.DataFrame({'ProfileID': test_profile_ids, 'RiskFlag': pred_20})
submission_20.to_csv('svm_20pct_submission.csv', index=False)

submission_full = pd.DataFrame({'ProfileID': test_profile_ids, 'RiskFlag': pred_full})
submission_full.to_csv('svm_full_submission.csv', index=False)

print("\nSubmission files created:")
print("✔ svm_20pct_submission.csv")
print("✔ svm_full_submission.csv")
