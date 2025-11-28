import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

print("--- SVM VIVA INFERENCE DEMO ---")

# --- 1. Load Artifacts ---
print("1. Loading Model and Preprocessors...")
try:
    model = joblib.load('svm_model.joblib')
    scaler = joblib.load('svm_scaler.joblib')
    train_cols = joblib.load('svm_columns.joblib')
    print("   [Success] SVM Model & Scaler loaded.")
except FileNotFoundError:
    print("   [Error] Artifacts missing! Run train_and_save.py first.")
    exit()

# --- 2. Load Test Data ---
print("2. Loading Test Data...")
try:
    df_test = pd.read_csv('viva_test_svm.csv')
    y_true = df_test['actual_spend_category']
    X_test_raw = df_test.drop('actual_spend_category', axis=1)
    print(f"   [Success] Loaded {len(df_test)} test rows.")
except FileNotFoundError:
    print("   [Error] 'viva_test_svm.csv' missing!")
    exit()

# --- 3. Preprocessing (Must match Training exactly) ---
print("3. Preprocessing Data...")

# Re-apply Mappings (Copy-pasted logic for consistency)
df_proc = X_test_raw.copy()
age_map = {'15-24': 1, '25-44': 2, '45-64': 3, '65+': 4, 'Missing': 0}
booked_map = {'0-7': 1, '8-14': 2, '15-30': 3, '31-60': 4, '61-90': 5, '90+': 6, 'Missing': 0}
days_map = {'1-3': 1, '4-6': 2, '7-14': 3, '15-30': 4, '30+': 5, 'Missing': 0}

for col in ['num_females', 'num_males']:
    if col in df_proc.columns: df_proc[col] = df_proc[col].fillna(df_proc[col].median())

if 'age_group' in df_proc.columns: df_proc['age_group'] = df_proc['age_group'].map(age_map).fillna(0).astype(int)
if 'days_booked_before_trip' in df_proc.columns: df_proc['days_booked_before_trip'] = df_proc['days_booked_before_trip'].map(booked_map).fillna(0).astype(int)
if 'total_trip_days' in df_proc.columns: df_proc['total_trip_days'] = df_proc['total_trip_days'].map(days_map).fillna(0).astype(int)

for col in df_proc.select_dtypes(include=['object']).columns:
    df_proc[col] = df_proc[col].fillna('Missing')

# One-Hot Encoding & Alignment
df_ohe = pd.get_dummies(df_proc, drop_first=False)
df_ohe = df_ohe.reindex(columns=train_cols, fill_value=0)

# Scaling (Using the loaded scaler)
X_test_final = pd.DataFrame(scaler.transform(df_ohe), columns=train_cols)

# --- 4. Prediction ---
print("4. Predicting...")
y_pred = model.predict(X_test_final)

# --- 5. Results ---
acc = accuracy_score(y_true, y_pred)

print("\n" + "="*40)
print(f" SVM MODEL ACCURACY: {acc:.2%}")
print("="*40)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))
print("-" * 40)
print("Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
print("="*40)