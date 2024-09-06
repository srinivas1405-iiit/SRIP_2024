import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
file_path = r'chunk2.csv'
data = pd.read_csv(file_path, low_memory=False)

# Identify the features and targets
features = [
    'source_id', 'source_origin_time', 'source_origin_uncertainty_sec', 'source_latitude',
    'source_longitude', 'source_error_sec', 'source_gap_deg', 'source_horizontal_uncertainty_km',
    'source_depth_km', 'source_depth_uncertainty_km', 'source_magnitude_type', 
    'source_magnitude_author', 'source_mechanism_strike_dip_rake', 'source_distance_deg', 
    'source_distance_km', 'back_azimuth_deg', 'snr_db', 'coda_end_sample', 
    'trace_start_time', 'trace_category', 'trace_name'
]

# Ensure all columns are present
assert all(col in data.columns for col in features + ['source_magnitude', 'p_arrival_sample', 's_arrival_sample']), "Some columns are missing!"

# Select the relevant columns
X = data[features]
y_magnitude = data['source_magnitude']
y_p_arrival = data['p_arrival_sample']
y_s_arrival = data['s_arrival_sample']

# Limit the number of unique values for high cardinality features (example for 'trace_name')
top_trace_names = X['trace_name'].value_counts().index[:100]  # Keep only the top 100 most frequent values
X.loc[~X['trace_name'].isin(top_trace_names), 'trace_name'] = 'other'

# Encode categorical features
X = pd.get_dummies(X, columns=['source_magnitude_type', 'trace_category', 'trace_name'], drop_first=True)

# Convert all columns to numeric, forcing non-convertible data to NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Fill NaN values with 0 or another appropriate value
X.fillna(0, inplace=True)

# Train-test split for each target
X_train, X_test, y_magnitude_train, y_magnitude_test = train_test_split(X, y_magnitude, test_size=0.2, random_state=42)
_, _, y_p_arrival_train, y_p_arrival_test = train_test_split(X, y_p_arrival, test_size=0.2, random_state=42)
_, _, y_s_arrival_train, y_s_arrival_test = train_test_split(X, y_s_arrival, test_size=0.2, random_state=42)

# Initialize and train preliminary models for feature importance
prelim_rf_model_magnitude = RandomForestRegressor(n_estimators=100, random_state=42)
prelim_rf_model_p_arrival = RandomForestRegressor(n_estimators=100, random_state=42)
prelim_rf_model_s_arrival = RandomForestRegressor(n_estimators=100, random_state=42)


prelim_rf_model_magnitude.fit(X_train, y_magnitude_train)
prelim_rf_model_p_arrival.fit(X_train, y_p_arrival_train)
prelim_rf_model_s_arrival.fit(X_train, y_s_arrival_train)

# Get feature importances
feature_importances_magnitude = prelim_rf_model_magnitude.feature_importances_
feature_importances_p_arrival = prelim_rf_model_p_arrival.feature_importances_
feature_importances_s_arrival = prelim_rf_model_s_arrival.feature_importances_

# Create importance dataframes
importance_df_magnitude = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances_magnitude})
importance_df_p_arrival = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances_p_arrival})
importance_df_s_arrival = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances_s_arrival})

# Plot feature importances
plt.figure(figsize=(15, 10))
sns.barplot(x='Importance', y='Feature', data=importance_df_magnitude.sort_values(by='Importance', ascending=False))
plt.title('Feature Importances for Magnitude')
plt.show()

plt.figure(figsize=(15, 10))
sns.barplot(x='Importance', y='Feature', data=importance_df_p_arrival.sort_values(by='Importance', ascending=False))
plt.title('Feature Importances for P Arrival Sample')
plt.show()

plt.figure(figsize=(15, 10))
sns.barplot(x='Importance', y='Feature', data=importance_df_s_arrival.sort_values(by='Importance', ascending=False))
plt.title('Feature Importances for S Arrival Sample')
plt.show()

# Select top features (e.g., top 10)
top_features_magnitude = importance_df_magnitude.sort_values(by='Importance', ascending=False).head(10)['Feature'].tolist()
top_features_p_arrival = importance_df_p_arrival.sort_values(by='Importance', ascending=False).head(10)['Feature'].tolist()
top_features_s_arrival = importance_df_s_arrival.sort_values(by='Importance', ascending=False).head(10)['Feature'].tolist()

# Train final models using top features
X_train_magnitude_top = X_train[top_features_magnitude]
X_test_magnitude_top = X_test[top_features_magnitude]

X_train_p_arrival_top = X_train[top_features_p_arrival]
X_test_p_arrival_top = X_test[top_features_p_arrival]

X_train_s_arrival_top = X_train[top_features_s_arrival]
X_test_s_arrival_top = X_test[top_features_s_arrival]

rf_model_magnitude = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_p_arrival = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_s_arrival = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model_magnitude.fit(X_train_magnitude_top, y_magnitude_train)
rf_model_p_arrival.fit(X_train_p_arrival_top, y_p_arrival_train)
rf_model_s_arrival.fit(X_train_s_arrival_top, y_s_arrival_train)

# Predictions
y_pred_magnitude_top = rf_model_magnitude.predict(X_test_magnitude_top)
y_pred_p_arrival_top = rf_model_p_arrival.predict(X_test_p_arrival_top)
y_pred_s_arrival_top = rf_model_s_arrival.predict(X_test_s_arrival_top)

# Evaluation for magnitude
mae_magnitude_top = mean_absolute_error(y_magnitude_test, y_pred_magnitude_top)
mse_magnitude_top = mean_squared_error(y_magnitude_test, y_pred_magnitude_top)
r_squared_magnitude_top = rf_model_magnitude.score(X_test_magnitude_top, y_magnitude_test)

print(f"Using Top Features for Magnitude:")
print(f"MAE: {mae_magnitude_top}")
print(f"MSE: {mse_magnitude_top}")
print(f"R-squared: {r_squared_magnitude_top}")
print(f"Top Features: {top_features_magnitude}")

# Evaluation for p_arrival_sample
mae_p_arrival_top = mean_absolute_error(y_p_arrival_test, y_pred_p_arrival_top)
mse_p_arrival_top = mean_squared_error(y_p_arrival_test, y_pred_p_arrival_top)
r_squared_p_arrival_top = rf_model_p_arrival.score(X_test_p_arrival_top, y_p_arrival_test)

print(f"Using Top Features for P Arrival Sample:")
print(f"MAE: {mae_p_arrival_top}")
print(f"MSE: {mse_p_arrival_top}")
print(f"R-squared: {r_squared_p_arrival_top}")
print(f"Top Features: {top_features_p_arrival}")

# Evaluation for s_arrival_sample
mae_s_arrival_top = mean_absolute_error(y_s_arrival_test, y_pred_s_arrival_top)
mse_s_arrival_top = mean_squared_error(y_s_arrival_test, y_pred_s_arrival_top)
r_squared_s_arrival_top = rf_model_s_arrival.score(X_test_s_arrival_top, y_s_arrival_test)

print(f"Using Top Features for S Arrival Sample:")
print(f"MAE: {mae_s_arrival_top}")
print(f"MSE: {mse_s_arrival_top}")
print(f"R-squared: {r_squared_s_arrival_top}")
print(f"Top Features: {top_features_s_arrival}")

# Plot actual vs predicted values for magnitude
plt.figure(figsize=(12, 8))
plt.scatter(y_magnitude_test, y_pred_magnitude_top, alpha=0.3)
plt.plot([y_magnitude_test.min(), y_magnitude_test.max()], [y_magnitude_test.min(), y_magnitude_test.max()], 'r--')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Actual vs Predicted Magnitudes (Top Features)')
plt.show()

# Plot actual vs predicted values for p_arrival_sample
plt.figure(figsize=(12, 8))
plt.scatter(y_p_arrival_test, y_pred_p_arrival_top, alpha=0.3)
plt.plot([y_p_arrival_test.min(), y_p_arrival_test.max()], [y_p_arrival_test.min(), y_p_arrival_test.max()], 'r--')
plt.xlabel('Actual P Arrival Sample')
plt.ylabel('Predicted P Arrival Sample')
plt.title('Actual vs Predicted P Arrival Samples (Top Features)')
plt.show()

# Plot actual vs predicted values for s_arrival_sample
plt.figure(figsize=(12, 8))
plt.scatter(y_s_arrival_test, y_pred_s_arrival_top, alpha=0.3)
plt.plot([y_s_arrival_test.min(), y_s_arrival_test.max()], [y_s_arrival_test.min(), y_s_arrival_test.max()], 'r--')
plt.xlabel('Actual S Arrival Sample')
plt.ylabel('Predicted S Arrival Sample')
plt.title('Actual vs Predicted S Arrival Samples (Top Features)')
plt.show()

# Function to take user input for predicting magnitude
def predict_magnitude():
    user_input = {}
    for feature in top_features_magnitude:
        value = input(f"Enter value for {feature}: ")
        try:
            user_input[feature] = float(value)
        except ValueError:
            user_input[feature] = value
    
    user_df = pd.DataFrame([user_input])
    
    for col in X_train_magnitude_top.columns:
        if col not in user_df.columns:
            user_df[col] = 0
    
    user_df = pd.get_dummies(user_df, drop_first=True)
    
    for col in X_train_magnitude_top.columns:
        if col not in user_df.columns:
            user_df[col] = 0
    
    user_df = user_df[X_train_magnitude_top.columns]
    
    prediction = rf_model_magnitude.predict(user_df)
    print(f"Predicted Magnitude: {prediction[0]}")

# Function to take user input for predicting p_arrival_sample
def predict_p_arrival_time():
    user_input = {}
    for feature in top_features_p_arrival:
        value = input(f"Enter value for {feature}: ")
        try:
            user_input[feature] = float(value)
        except ValueError:
            user_input[feature] = value
    
    user_df = pd.DataFrame([user_input])
    
    for col in X_train_p_arrival_top.columns:
        if col not in user_df.columns:
            user_df[col] = 0
    
    user_df = pd.get_dummies(user_df, drop_first=True)
    
    for col in X_train_p_arrival_top.columns:
        if col not in user_df.columns:
            user_df[col] = 0
    
    user_df = user_df[X_train_p_arrival_top.columns]
    
    prediction = rf_model_p_arrival.predict(user_df)
    # print(f"Predicted P Arrival Sample: {prediction[0]}")

    sampling_rate = 100  # in Hz (samples per second)

    # As we have the p_arrival_sample
    p_arrival_sample = prediction[0]

    # Calculating p_arrival_time
    p_arrival_time = p_arrival_sample / sampling_rate
    print(f"Predicted P Arrival Time (in sec): {p_arrival_time}")


# Function to take user input for predicting s_arrival_sample
def predict_s_arrival_time():
    user_input = {}
    for feature in top_features_s_arrival:
        value = input(f"Enter value for {feature}: ")
        try:
            user_input[feature] = float(value)
        except ValueError:
            user_input[feature] = value
    
    user_df = pd.DataFrame([user_input])
    
    for col in X_train_s_arrival_top.columns:
        if col not in user_df.columns:
            user_df[col] = 0
    
    user_df = pd.get_dummies(user_df, drop_first=True)
    
    for col in X_train_s_arrival_top.columns:
        if col not in user_df.columns:
            user_df[col] = 0
    
    user_df = user_df[X_train_s_arrival_top.columns]
    
    prediction = rf_model_s_arrival.predict(user_df)
    # print(f"Predicted S Arrival Sample: {prediction[0]}")

    sampling_rate = 100  # in Hz (samples per second)

    # As we have the s_arrival_sample
    s_arrival_sample = prediction[0]

    # Calculating p_arrival_time and s_arrival_time
    s_arrival_time = s_arrival_sample / sampling_rate

    print(f"Predicted S Arrival Time(in sec): {s_arrival_time}")


# Example usage:
predict_magnitude()
predict_p_arrival_time()
predict_s_arrival_time()
