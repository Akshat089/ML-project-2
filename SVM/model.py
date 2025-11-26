import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
import numpy as np
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. Load Data ---
print("Loading data...")
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Error: One or both files not found. Ensure 'train.csv' and 'test.csv' are in the same directory.")
    exit()

# Clean Target
df_train.dropna(subset=['spend_category'], inplace=True)
df_train['spend_category'] = df_train['spend_category'].astype(int)

# Store trip_id for test set submission
test_trip_ids = df_test['trip_id']

# Separate features (X) and target (y)
X_train = df_train.drop(['trip_id', 'spend_category'], axis=1)
y_train = df_train['spend_category']
X_test = df_test.drop('trip_id', axis=1)

# --- 2. Feature Mapping and Imputation (FIXED) ---
print("Handling missing values and applying Ordinal Encoding...")

# Define Ordinal Mappings 
age_map = {'15-24': 1, '25-44': 2, '45-64': 3, '65+': 4, 'Missing': 0}
booked_map = {'0-7': 1, '8-14': 2, '15-30': 3, '31-60': 4, '61-90': 5, '90+': 6, 'Missing': 0}
days_map = {'1-3': 1, '4-6': 2, '7-14': 3, '15-30': 4, '30+': 5, 'Missing': 0}

ordinal_cols = ['age_group', 'days_booked_before_trip', 'total_trip_days']
map_list = [age_map, booked_map, days_map]

# Imputation and Ordinal Encoding
for df in [X_train, X_test]:
    
    # Handle Numerical (Float) Imputation 
    float_cols = ['num_females', 'num_males']
    for col in float_cols:
        median_val = df_train[col].median()
        df[col] = df[col].fillna(median_val)

    # Impute and Map Ordinal Columns (FIXED)
    for col, mapping in zip(ordinal_cols, map_list):
        df[col] = df[col].fillna('Missing')
        # Use fillna(0) to safely handle any unmapped/missing values before casting to int
        df[col] = df[col].map(mapping).fillna(0).astype(int)
    
    # Impute Remaining Nominal Categorical Columns 
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Missing')

# --- 3. One-Hot Encoding and Scaling ---
print("Applying One-Hot Encoding and Scaling...")

# Identify remaining object columns for OHE
object_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Combine for consistent One-Hot Encoding
X_combined = pd.concat([X_train, X_test], ignore_index=True)
X_combined_ohe = pd.get_dummies(X_combined, columns=object_cols, drop_first=False)

# Split back and align features
X_train_processed = X_combined_ohe.iloc[:len(X_train)].copy()
X_test_processed = X_combined_ohe.iloc[len(X_train):].copy()
X_test_processed = X_test_processed.reindex(columns=X_train_processed.columns, fill_value=0)

# Identify all numerical columns 
numeric_cols = X_train_processed.select_dtypes(include=[np.number]).columns.tolist()

# Apply scaling to all numerical columns
scaler = StandardScaler()
X_train_processed[numeric_cols] = scaler.fit_transform(X_train_processed[numeric_cols])
X_test_processed[numeric_cols] = scaler.transform(X_test_processed[numeric_cols])

# --- 4. Model Tuning (Randomized Search CV) ---
print("\nStarting Randomized Search for Optimal C and Gamma...")
search_start = time.time()

# Define the model base
svc_model = SVC(
    kernel='rbf',
    class_weight='balanced', # Essential for imbalance
    random_state=42
)

# Define the log-uniform parameter distribution for C and gamma
param_distributions = {
    # C: 0.1 to 1000
    'C': loguniform(0.1, 1000), 
    
    # gamma: 0.0001 to 10
    'gamma': loguniform(0.0001, 10) 
}

# Setup the Randomized Search
random_search = RandomizedSearchCV(
    estimator=svc_model,
    param_distributions=param_distributions,
    n_iter=50,       # Number of parameter settings to sample
    scoring='f1_weighted', # Metric for optimization
    cv=3,            # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1        # Use all CPU cores
)

random_search.fit(X_train_processed, y_train) 
search_end = time.time()

print("--- Randomized Search Results ---")
print(f"Optimal C: {random_search.best_params_['C']:.4f}")
print(f"Optimal Gamma: {random_search.best_params_['gamma']:.4f}")
print(f"Best CV Score (F1-weighted): {random_search.best_score_:.4f}")
print(f"Total Search Time: {(search_end - search_start) / 60:.2f} minutes.")


# --- 5. Final Prediction and Submission ---
print("\nTraining final model and generating predictions...")

# Use the best model found by the search
final_model = random_search.best_estimator_
predictions = final_model.predict(X_test_processed)

submission_df = pd.DataFrame({
    'trip_id': test_trip_ids,
    'spend_category': predictions.astype(int)
})

submission_filename = 'svm_rbf_randomsearch_submission.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\nSubmission file created successfully: {submission_filename}")