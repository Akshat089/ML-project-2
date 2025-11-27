import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load Data ---
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Clean Target
df_train.dropna(subset=['spend_category'], inplace=True)
df_train['spend_category'] = df_train['spend_category'].astype(int)
y_train = df_train['spend_category']
test_trip_ids = df_test['trip_id']

# Drop ID
X_train = df_train.drop(['trip_id', 'spend_category'], axis=1)
X_test = df_test.drop('trip_id', axis=1)

# --- 2. Advanced Feature Engineering Function ---
def engineer_features(df):
    df = df.copy()
    
    # Fill numeric NaNs
    df['num_females'] = df['num_females'].fillna(0)
    df['num_males'] = df['num_males'].fillna(0)
    df['mainland_stay_nights'] = df['mainland_stay_nights'].fillna(0)
    df['island_stay_nights'] = df['island_stay_nights'].fillna(0)
    
    # 1. Totals
    df['group_size'] = df['num_females'] + df['num_males']
    df['total_nights'] = df['mainland_stay_nights'] + df['island_stay_nights']
    
    # 2. Interaction: Person-Nights (Crucial for cost)
    df['person_nights'] = df['group_size'] * df['total_nights']
    
    # 3. Ratio: Female Ratio (Demographics often impact spending types)
    # Adding small epsilon to avoid div by zero
    df['female_ratio'] = df['num_females'] / (df['group_size'] + 0.001)
    
    # 4. Log Transforms on skewed continuous data
    for col in ['total_nights', 'person_nights', 'group_size']:
        df[col] = np.log1p(df[col])
        
    return df

X_train = engineer_features(X_train)
X_test = engineer_features(X_test)

# --- 3. Column Definitions ---
# We treat Ordinals as Categorical now (Unchaining)
categorical_features = [
    'country', 'age_group', 'travel_companions', 'main_activity', 'visit_purpose',
    'is_first_visit', 'tour_type', 'info_source', 'days_booked_before_trip',
    'arrival_weather', 'total_trip_days', 'has_special_requirements'
]

binary_features = [
    'intl_transport_included', 'accomodation_included', 'food_included', 
    'domestic_transport_included', 'sightseeing_included', 'guide_included', 
    'insurance_included'
]

numeric_features = ['group_size', 'total_nights', 'person_nights', 'female_ratio']

# --- 4. Building the Preprocessing Pipeline ---

# Numeric Pipeline: Impute -> Polynomials -> Scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    # Degree 2 creates interactions like "GroupSize * FemaleRatio"
    ('poly', PolynomialFeatures(degree=2, include_bias=False)), 
    ('scaler', StandardScaler())
])

# Categorical Pipeline: Impute -> OneHot
# handle_unknown='ignore' prevents crashing on new categories in test data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Binary Pipeline: Impute -> Identity
binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # We map Yes/No inside the pipeline using OneHot drop='if_binary' effectively
    ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('bin', binary_transformer, binary_features)
    ])

# --- 5. The Voting Ensemble ---
# We combine 3 Logistic Regressions with different mathematical solvers

clf1 = LogisticRegression(
    solver='lbfgs', multi_class='multinomial', 
    C=1.0, max_iter=2000, class_weight='balanced', random_state=42
)

clf2 = LogisticRegression(
    solver='saga', penalty='l1', multi_class='multinomial', # L1 Lasso
    C=0.5, max_iter=2000, class_weight='balanced', random_state=42
)

clf3 = LogisticRegression(
    solver='saga', penalty='l2', multi_class='multinomial', # L2 Ridge (Stronger Reg)
    C=0.1, max_iter=2000, class_weight='balanced', random_state=42
)

# Soft Voting averages the probabilities (Proba) of all models
voting_clf = VotingClassifier(
    estimators=[('lr_standard', clf1), ('lr_lasso', clf2), ('lr_ridge', clf3)],
    voting='soft',
    n_jobs=-1
)

# Create Final Pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', voting_clf)])

# --- 6. Train and Predict ---
print("Training Voting Ensemble...")
model_pipeline.fit(X_train, y_train)

print("Generating predictions...")
predictions = model_pipeline.predict(X_test)

submission = pd.DataFrame({'trip_id': test_trip_ids, 'spend_category': predictions.astype(int)})
submission.to_csv('polynomial_voting_submission.csv', index=False)
print("Done. Saved 'polynomial_voting_submission.csv'")