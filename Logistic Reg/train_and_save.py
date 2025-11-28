import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import warnings

warnings.filterwarnings('ignore')

# --- 1. Define Feature Engineering Function ---
# (Must be defined here to train)
def engineer_features(df):
    df = df.copy()
    df['num_females'] = df['num_females'].fillna(0)
    df['num_males'] = df['num_males'].fillna(0)
    df['mainland_stay_nights'] = df['mainland_stay_nights'].fillna(0)
    df['island_stay_nights'] = df['island_stay_nights'].fillna(0)
    
    df['group_size'] = df['num_females'] + df['num_males']
    df['total_nights'] = df['mainland_stay_nights'] + df['island_stay_nights']
    df['person_nights'] = df['group_size'] * df['total_nights']
    df['female_ratio'] = df['num_females'] / (df['group_size'] + 0.001)
    
    for col in ['total_nights', 'person_nights', 'group_size']:
        df[col] = np.log1p(df[col])
    return df

# --- 2. Load and Split Data ---
print("Loading data...")
df_full = pd.read_csv('train.csv')
df_full.dropna(subset=['spend_category'], inplace=True)
df_full['spend_category'] = df_full['spend_category'].astype(int)

X = df_full.drop(['trip_id', 'spend_category'], axis=1)
y = df_full['spend_category']

# Create a 20% split for the VIVA demonstration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Build Pipeline ---
X_train_eng = engineer_features(X_train)

categorical_features = ['country', 'age_group', 'travel_companions', 'main_activity', 'visit_purpose', 'is_first_visit', 'tour_type', 'info_source', 'days_booked_before_trip', 'arrival_weather', 'total_trip_days', 'has_special_requirements']
binary_features = ['intl_transport_included', 'accomodation_included', 'food_included', 'domestic_transport_included', 'sightseeing_included', 'guide_included', 'insurance_included']
numeric_features = ['group_size', 'total_nights', 'person_nights', 'female_ratio']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
    ('bin', binary_transformer, binary_features)
])

# Voting Ensemble
clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1.0, max_iter=2000, class_weight='balanced', random_state=42)
clf2 = LogisticRegression(solver='saga', penalty='l1', multi_class='multinomial', C=0.5, max_iter=2000, class_weight='balanced', random_state=42)
clf3 = LogisticRegression(solver='saga', penalty='l2', multi_class='multinomial', C=0.1, max_iter=2000, class_weight='balanced', random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('lr_standard', clf1), 
    ('lr_lasso', clf2), 
    ('lr_ridge', clf3)], 
    voting='soft', n_jobs=-1)

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', voting_clf)])

# --- 4. Train and Save ---
print("Training model (this might take a moment)...")
model_pipeline.fit(X_train_eng, y_train)

print("Saving model artifact to 'voting_model.joblib'...")
joblib.dump(model_pipeline, 'voting_model.joblib')

print("Saving test dataset to 'viva_test_data.csv'...")
# We save X_test AND y_test together so we can check accuracy during viva
test_set = X_test.copy()
test_set['actual_spend_category'] = y_test
test_set.to_csv('viva_test_data.csv', index=False)

print("\nSUCCESS! You are ready for the Viva.")
print("Files created: 'voting_model.joblib', 'viva_test_data.csv'")