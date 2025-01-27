import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

# Preprocess data
def preprocess_data(df):
    """Preprocess the data: handle missing values, encode categorical features, scale numeric features."""
    df = df.dropna()  # Remove rows with missing values
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # One-hot encoding

    # Scale numerical features
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

# Split data into features and target
def split_features_target(df, target_column):
    """Split the DataFrame into features (X) and target (y)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

# Train models
def train_models(X_train, y_train):
    """Train Random Forest and Gradient Boosting models."""
    models = {}

    # Random Forest Regressor
    rf = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10, min_samples_split=5)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    # Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=5)
    gbr.fit(X_train, y_train)
    models['GradientBoosting'] = gbr

    return models

# Evaluate models
def evaluate_models(models, X_test, y_test):
    """Evaluate models on the test set."""
    for name, model in models.items():
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"{name} Performance:")
        print(f"  Mean Squared Error: {mse:.2f}")
        print(f"  R^2 Score: {r2:.2f}\n")

# Main function
def main():
    # File path to the dataset
    file_path = 'Housing.csv'  # Replace with your dataset path

    # Load and preprocess data
    df = load_data(file_path)
    df = preprocess_data(df)

    # Split data into training and testing sets
    target_column = 'price'  # Replace with the target column in your dataset
    X, y = split_features_target(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate models
    evaluate_models(models, X_test, y_test)

if __name__ == '__main__':
    main()