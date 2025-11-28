import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import loguniform
import warnings
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
print("Loading data...")
try:
    df_train = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: 'train.csv' not found.")
    exit()

# Clean Target
df_train.dropna(subset=['spend_category'], inplace=True)
df_train['spend_category'] = df_train['spend_category'].astype(int)

# Split features and target
X = df_train.drop(['trip_id', 'spend_category'], axis=1)
y = df_train['spend_category']

# Create a local train/test split for VIVA verification
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Feature Mapping and Preprocessing Function ---
# We define this as a function so we can reuse it in the viva script if needed, 
# though for SVM we usually process everything before the model.
def preprocess_data(df, is_training=True, train_columns=None, fitted_scaler=None):
    df = df.copy()
    
    # Mappings
    age_map = {'15-24': 1, '25-44': 2, '45-64': 3, '65+': 4, 'Missing': 0}
    booked_map = {'0-7': 1, '8-14': 2, '15-30': 3, '31-60': 4, '61-90': 5, '90+': 6, 'Missing': 0}
    days_map = {'1-3': 1, '4-6': 2, '7-14': 3, '15-30': 4, '30+': 5, 'Missing': 0}
    
    # Impute Numerical
    for col in ['num_females', 'num_males']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Map Ordinals
    if 'age_group' in df.columns: df['age_group'] = df['age_group'].map(age_map).fillna(0).astype(int)
    if 'days_booked_before_trip' in df.columns: df['days_booked_before_trip'] = df['days_booked_before_trip'].map(booked_map).fillna(0).astype(int)
    if 'total_trip_days' in df.columns: df['total_trip_days'] = df['total_trip_days'].map(days_map).fillna(0).astype(int)

    # Impute remaining Categoricals
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Missing')

    # One-Hot Encoding
    df_ohe = pd.get_dummies(df, drop_first=False)
    
    # Align Columns
    if is_training:
        final_cols = df_ohe.columns.tolist()
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_ohe), columns=final_cols)
        return df_scaled, final_cols, scaler
    else:
        # Align columns to match training
        df_ohe = df_ohe.reindex(columns=train_columns, fill_value=0)
        df_scaled = pd.DataFrame(fitted_scaler.transform(df_ohe), columns=train_columns)
        return df_scaled

# --- 3. Process Training Data ---
print("Preprocessing Training Data...")
X_train_processed, train_cols, scaler_obj = preprocess_data(X_train_raw, is_training=True)

# --- 4. Model Tuning (Randomized Search) ---
print("\nStarting Randomized Search (this may take time)...")
svc_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)

param_distributions = {
    'C': loguniform(0.1, 1000), 
    'gamma': loguniform(0.0001, 10) 
}

random_search = RandomizedSearchCV(
    estimator=svc_model,
    param_distributions=param_distributions,
    n_iter=20, # Reduced iterations slightly for speed
    scoring='f1_weighted',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_processed, y_train)

print(f"Best Params: {random_search.best_params_}")

# --- 5. Save Artifacts ---
print("Saving artifacts...")

# Save the Best Estimator
joblib.dump(random_search.best_estimator_, 'svm_model.joblib')

# Save Scaler and Columns (Required for preprocessing test data)
joblib.dump(scaler_obj, 'svm_scaler.joblib')
joblib.dump(train_cols, 'svm_columns.joblib')

# Save Test Data for VIVA
X_test_raw['actual_spend_category'] = y_test
X_test_raw.to_csv('viva_test_svm.csv', index=False)

print("\nReady for Viva. Files created:")
print("- svm_model.joblib")
print("- svm_scaler.joblib")
print("- svm_columns.joblib")
print("- viva_test_svm.csv")