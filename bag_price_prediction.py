import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def clean_outliers(df):
    """Remove or clip extreme outliers"""
    df = df.copy()
    
    # Clip Weight Capacity outliers
    q1 = df['Weight Capacity (kg)'].quantile(0.01)
    q3 = df['Weight Capacity (kg)'].quantile(0.99)
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].clip(q1, q3)
    
    # Clip Compartments to reasonable range (1-20)
    df['Compartments'] = df['Compartments'].clip(1, 20)
    
    # Remove price outliers if price column exists
    if 'Price' in df.columns:
        price_q1 = df['Price'].quantile(0.01)
        price_q3 = df['Price'].quantile(0.99)
        df = df[df['Price'].between(price_q1, price_q3)]
    
    return df

def create_features(df):
    """Create new features"""
    df = df.copy()
    
    # Size encoding
    size_map = {'Small': 1, 'Medium': 2, 'Large': 3}
    df['size_numeric'] = df['Size'].map(size_map)
    
    # Premium features count
    df['premium_features'] = (df['Laptop Compartment'].map({'Yes': 1, 'No': 0}) + 
                            df['Waterproof'].map({'Yes': 1, 'No': 0}))
    
    # Compartment density (compartments per size)
    df['compartment_density'] = df['Compartments'] / df['size_numeric']
    
    # Brand-Material combination
    df['brand_material'] = df['Brand'] + '_' + df['Material']
    
    # Style-Size combination
    df['style_size'] = df['Style'] + '_' + df['Size']
    
    # Material category (synthetic vs natural)
    synthetic_materials = ['Nylon', 'Polyester']
    df['is_synthetic'] = df['Material'].isin(synthetic_materials).astype(int)
    
    # Color category (dark vs light)
    dark_colors = ['Black', 'Navy', 'Brown']
    light_colors = ['White', 'Beige', 'Yellow']
    df['color_category'] = df['Color'].apply(lambda x: 
                                           'dark' if x in dark_colors else
                                           'light' if x in light_colors else 'medium')
    
    # Weight capacity bins
    df['weight_capacity_bins'] = pd.qcut(df['Weight Capacity (kg)'], q=5, labels=['VL', 'L', 'M', 'H', 'VH'])
    
    return df

def preprocess_data(df, encoders=None, train=True):
    """Preprocess the data"""
    df = df.copy()
    
    # Clean outliers
    df = clean_outliers(df)
    
    # Basic cleaning
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')
        df[col] = df[col].astype(str).str.strip()
    
    # Create new features
    df = create_features(df)
    
    # Convert binary features
    binary_map = {'Yes': 1, 'No': 0, 'Unknown': -1}
    df['Laptop Compartment'] = df['Laptop Compartment'].map(binary_map)
    df['Waterproof'] = df['Waterproof'].map(binary_map)
    
    # Categorical features to encode
    cat_features = ['Brand', 'Material', 'Size', 'Style', 'Color', 
                   'brand_material', 'style_size', 'color_category', 
                   'weight_capacity_bins']
    
    if train:
        encoders = {}
        for feature in cat_features:
            if feature in df.columns:  # Check if feature exists
                encoders[feature] = LabelEncoder()
                df[feature] = encoders[feature].fit_transform(df[feature])
        return df, encoders
    else:
        for feature in cat_features:
            if feature in df.columns and feature in encoders:
                # Handle categories not seen during training
                known_categories = set(encoders[feature].classes_)
                df[feature] = df[feature].map(lambda x: 'Unknown' if x not in known_categories else x)
                df[feature] = encoders[feature].transform(df[feature])
        return df

# Load the datasets
print("Loading datasets...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_extra_df = pd.read_csv('training_extra.csv')

print("\nCleaning and preprocessing data...")
train_processed, encoders = preprocess_data(train_df, train=True)
train_extra_processed = preprocess_data(train_extra_df, encoders=encoders, train=False)
test_processed = preprocess_data(test_df, encoders=encoders, train=False)

# Combine training data
full_train = pd.concat([train_processed, train_extra_processed], axis=0, ignore_index=True)
print("Combined training set shape:", full_train.shape)

# Prepare features
feature_columns = ['Brand', 'Material', 'size_numeric', 'Compartments',
                  'Laptop Compartment', 'Waterproof', 'Style', 'Color',
                  'Weight Capacity (kg)', 'compartment_density', 'premium_features',
                  'brand_material', 'style_size', 'is_synthetic', 'color_category']

X = full_train[feature_columns]
y = full_train['Price']
X_test = test_processed[feature_columns]

# Create pipeline with scaling
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=200, 
                                  max_depth=15,
                                  min_samples_split=5,
                                  min_samples_leaf=2,
                                  n_jobs=-1,
                                  random_state=42))
])

# Perform cross-validation
print("\nPerforming cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_root_mean_squared_error')
print("Cross-validation RMSE scores:", -cv_scores)
print("Average RMSE:", -cv_scores.mean())
print("RMSE std:", cv_scores.std())

# Train final model
print("\nTraining final model...")
pipeline.fit(X, y)

# Make predictions
print("Making predictions...")
test_predictions = pipeline.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Price': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created!")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': pipeline.named_steps['model'].feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Save feature importance to CSV
feature_importance.to_csv('feature_importance.csv', index=False)

print("\nSaving model and encoders...")
import pickle

# Save the final model
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Save the encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("Model and encoders saved successfully!")
