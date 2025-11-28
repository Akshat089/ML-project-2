import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# --- 1. Re-define Feature Engineering ---
# (Required because joblib needs the function definition to exist to load the object)
def engineer_features(df):
    df = df.copy()
    # Handle NaNs
    df['num_females'] = df['num_females'].fillna(0)
    df['num_males'] = df['num_males'].fillna(0)
    df['mainland_stay_nights'] = df['mainland_stay_nights'].fillna(0)
    df['island_stay_nights'] = df['island_stay_nights'].fillna(0)
    
    # Create Features
    df['group_size'] = df['num_females'] + df['num_males']
    df['total_nights'] = df['mainland_stay_nights'] + df['island_stay_nights']
    df['person_nights'] = df['group_size'] * df['total_nights']
    df['female_ratio'] = df['num_females'] / (df['group_size'] + 0.001)
    
    # Log Transforms
    for col in ['total_nights', 'person_nights', 'group_size']:
        df[col] = np.log1p(df[col])
    return df

print("--- VIVA INFERENCE DEMO ---")
print("1. Loading Pre-trained Model...")

# Load the model you saved earlier
try:
    model = joblib.load('voting_model.joblib')
    print("   [OK] Model Loaded Successfully.")
except FileNotFoundError:
    print("   [ERROR] 'voting_model.joblib' not found! Run train_and_save.py first.")
    exit()

print("2. Loading Test Data...")
try:
    # Load the specific test split created for viva
    df_test = pd.read_csv('viva_test_data.csv')
    
    # Separate Features (X) and Labels (y)
    y_true = df_test['actual_spend_category']
    X_test_raw = df_test.drop('actual_spend_category', axis=1)
    
    print(f"   [OK] Loaded {len(df_test)} rows for testing.")
except FileNotFoundError:
    print("   [ERROR] 'viva_test_data.csv' not found! Run train_and_save.py first.")
    exit()

# --- 3. Preprocessing (Live) ---
print("3. Applying Feature Engineering...")
X_test_processed = engineer_features(X_test_raw)

# --- 4. Prediction ---
print("4. Running Predictions...")
y_pred = model.predict(X_test_processed)

# --- 5. Metrics Calculation ---
print("\n" + "="*40)
print("FINAL RESULTS")
print("="*40)

acc = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {acc:.2%}")
print("-" * 40)
print("Classification Report:\n")
print(classification_report(y_true, y_pred))
print("-" * 40)
print("Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
print("="*40)