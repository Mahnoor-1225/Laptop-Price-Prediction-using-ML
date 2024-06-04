import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
file_path = r'C:\Users\mahno\OneDrive\Desktop\internship\laptopPrice.csv'
df = pd.read_csv(file_path)

# Data Exploration
print("First few rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())
print("\nColumns in the dataset:")
print(df.columns)

# Check if expected columns exist and handle accordingly
expected_columns = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 
                    'ram_type', 'ssd', 'hdd', 'os', 'os_bit', 'graphic_card_gb', 'weight', 
                    'warranty', 'Touchscreen', 'msoffice', 'Price', 'rating', 
                    'Number of Ratings', 'Number of Reviews']
for column in expected_columns:
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in the dataset.")

# Feature Engineering
# Convert categorical features to numerical, if they exist
categorical_columns = ['brand', 'processor_brand', 'processor_name', 'ram_type', 'os', 'Touchscreen', 'msoffice']
for column in categorical_columns:
    if column in df.columns:
        df[column] = df[column].astype('category').cat.codes

# Handle 'processor_gnrtn' to convert it to numerical
if 'processor_gnrtn' in df.columns:
    df['processor_gnrtn'] = df['processor_gnrtn'].str.extract(r'(\d+)')
    df['processor_gnrtn'] = pd.to_numeric(df['processor_gnrtn'], errors='coerce').fillna(0).astype(int)

# Ensure numerical columns are in correct dtype
numerical_columns = ['ram_gb', 'ssd', 'hdd', 'os_bit', 'graphic_card_gb', 'weight', 'warranty', 'rating']
for column in numerical_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)

# Select features and target variable, ensuring the columns exist
feature_columns = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 
                   'ram_type', 'ssd', 'hdd', 'os', 'os_bit', 'graphic_card_gb', 'weight', 
                   'warranty', 'Touchscreen', 'msoffice', 'rating', 'Number of Ratings', 'Number of Reviews']
X = df[feature_columns]
if 'Price' in df.columns:
    y = df['Price']
else:
    raise KeyError("The target variable 'Price' is not found in the dataset.")

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
# Using Random Forest and Gradient Boosting Regressors
rf = RandomForestRegressor(random_state=42)
gbr = GradientBoostingRegressor(random_state=42)

# Train the models
rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)

# Model Evaluation
# Predicting on the test set
y_pred_rf = rf.predict(X_test)
y_pred_gbr = gbr.predict(X_test)

# Calculating evaluation metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))

print(f'Random Forest - MAE: {mae_rf}, RMSE: {rmse_rf}')
print(f'Gradient Boosting - MAE: {mae_gbr}, RMSE: {rmse_gbr}')

# Cross-Validation for model robustness
cv_scores_rf = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_scores_gbr = cross_val_score(gbr, X, y, cv=5, scoring='neg_mean_absolute_error')

print(f'Random Forest - Cross-Validated MAE: {-cv_scores_rf.mean()}')
print(f'Gradient Boosting - Cross-Validated MAE: {-cv_scores_gbr.mean()}')

# Plotting feature importances
if hasattr(rf, 'feature_importances_'):
    plt.figure(figsize=(10, 5))
    plt.barh(X.columns, rf.feature_importances_)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importances')
    plt.show()

if hasattr(gbr, 'feature_importances_'):
    plt.figure(figsize=(10, 5))
    plt.barh(X.columns, gbr.feature_importances_)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Gradient Boosting Feature Importances')
    plt.show()
