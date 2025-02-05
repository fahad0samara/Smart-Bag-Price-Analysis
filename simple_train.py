import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import os

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    print("Loading data...")
    # Load only the main training data (not the extra data)
    train_df = pd.read_csv(os.path.join(current_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(current_dir, 'test.csv'))

    # Use only 10% of training data for faster processing
    train_df = train_df.sample(n=30000, random_state=42)

    print("Processing data...")
    def process_data(df, encoders=None, imputers=None, train=True):
        df = df.copy()
        
        # Basic cleaning
        for col in ['Brand', 'Material', 'Size', 'Style', 'Color']:
            df[col] = df[col].fillna('Unknown')
        
        # Simple binary encoding
        df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0}).fillna(-1)
        df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0}).fillna(-1)
        
        # Simple size encoding
        df['size_numeric'] = df['Size'].map({'Small': 1, 'Medium': 2, 'Large': 3, 'Unknown': 0})
        
        # Basic features only
        df['premium_features'] = df['Laptop Compartment'] + df['Waterproof']
        
        # Fill NaN in numeric columns with median
        df['Compartments'] = df['Compartments'].fillna(df['Compartments'].median())
        df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].median())
        
        # Encode categorical features
        cat_features = ['Brand', 'Material', 'Style', 'Color']
        
        if train:
            encoders = {}
            imputers = {}
            for col in cat_features:
                encoders[col] = LabelEncoder()
                df[col] = encoders[col].fit_transform(df[col])
            
            # Create imputers for numerical features
            for col in ['Weight Capacity (kg)', 'Compartments']:
                imputers[col] = SimpleImputer(strategy='median')
                df[col] = imputers[col].fit_transform(df[[col]])[:,0]
            
            return df, encoders, imputers
        else:
            for col in cat_features:
                df[col] = df[col].map(lambda x: 'Unknown' if x not in encoders[col].classes_ else x)
                df[col] = encoders[col].transform(df[col])
            
            # Use the same imputers for numerical features
            for col in ['Weight Capacity (kg)', 'Compartments']:
                df[col] = imputers[col].transform(df[[col]])[:,0]
            
            return df

    # Process data
    train_processed, encoders, imputers = process_data(train_df, train=True)
    test_processed = process_data(test_df, encoders=encoders, imputers=imputers, train=False)

    # Select features
    features = ['Brand', 'Material', 'size_numeric', 'Compartments',
               'Laptop Compartment', 'Waterproof', 'Style', 'Color',
               'Weight Capacity (kg)', 'premium_features']

    X = train_processed[features]
    y = train_processed['Price']
    X_test = test_processed[features]

    print("Training model...")
    # Use Ridge Regression (simple and fast)
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    print("Making predictions...")
    test_predictions = model.predict(X_test)

    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Price': test_predictions
    })
    submission.to_csv(os.path.join(current_dir, 'submission.csv'), index=False)

    # Save model and encoders for the web app
    print("Saving model and encoders...")
    model_path = os.path.join(current_dir, 'model.pkl')
    encoders_path = os.path.join(current_dir, 'encoders.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(encoders_path, 'wb') as f:
        pickle.dump({'encoders': encoders, 'imputers': imputers}, f)

    print("Done! Model and encoders saved successfully!")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    print(traceback.format_exc())
