import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df, categorical_features, numeric_features):
    """Preprocess the dataset by one-hot encoding categorical features."""
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    features = numeric_features + list(df.columns.difference(numeric_features + ['SalePrice']))
    return df[features], df['SalePrice']

def impute_and_scale(X_train):
    """Impute missing values and scale the features."""
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_train)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, imputer, scaler

def evaluate_model(model, X_train_scaled, y_train):
    """Evaluate the model using cross-validation."""
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
    print(f"Cross-Validated MAE: {-np.mean(cv_scores):,.2f}")

def prepare_test_data(test_df, categorical_features, train_columns):
    """Prepare the test dataset with one-hot encoding and reindexing."""
    test_df = pd.get_dummies(test_df, columns=categorical_features, drop_first=True)
    test_df = test_df.reindex(columns=train_columns, fill_value=0)
    return test_df

def make_predictions(model, X_test_scaled, imputer, scaler):
    """Make predictions on the test dataset."""
    y_test_pred = model.predict(X_test_scaled)
    return np.maximum(y_test_pred, 0)  # Set negative predictions to zero

def save_predictions(test_df, predictions, submission_file_path):
    """Save predictions to a CSV file."""
    submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': predictions})
    submission_df.to_csv(submission_file_path, index=False)
    print(f'Predictions have been saved to {submission_file_path}')

def main(train_file_path, test_file_path, submission_file_path):
    # Load training data
    train_df = load_data(train_file_path)

    # Define features
    categorical_features = [
        'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
        'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
    ]
    
    numeric_features = [
        'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd',
        'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
        'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
        'MoSold', 'YrSold'
    ]

    # Preprocess training data
    X_train, y_train = preprocess_data(train_df, categorical_features, numeric_features)
    
    # Impute and scale training data
    X_train_scaled, imputer, scaler = impute_and_scale(X_train)

    # Create and evaluate the model
    model = LinearRegression()
    evaluate_model(model, X_train_scaled, y_train)

    # Fit the model
    model.fit(X_train_scaled, y_train)

    # Load and prepare test data
    test_df = load_data(test_file_path)
    test_df = prepare_test_data(test_df, categorical_features, X_train.columns)

    # Impute and scale test data
    X_test = test_df[X_train.columns]
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Make predictions
    y_test_pred = make_predictions(model, X_test_scaled, imputer, scaler)

    # Save predictions
    save_predictions(test_df, y_test_pred, submission_file_path)

# Paths to the datasets
train_file_path = '/content/train.csv'  # Update this path
test_file_path = '/content/test.csv'  # Update this path
submission_file_path = '/content/submission.csv'  # Update this path

# Run the main function
main(train_file_path, test_file_path, submission_file_path)
