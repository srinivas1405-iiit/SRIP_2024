import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load your dataset
data = pd.read_csv("Portland_RotD50_Vs30ZX.csv")

# Define the threshold for strong motion
threshold = 0.005

# Define the time steps for the SA columns
time_steps = [0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200, 0.250, 0.300, 0.400, 0.500, 
              0.750, 1.000, 1.500, 2.000, 3.000, 4.000, 5.000, 7.500, 10.000]

# Calculate the duration of strong motion
def calculate_strong_motion_duration(row, threshold, time_steps):
    # Extract SA values from the row and ensure they are numeric
    sa_values = pd.to_numeric(row.filter(regex=r'^SA\('), errors='coerce').values
    time_steps_filtered = [time_steps[i] for i in range(len(sa_values))]
    
    # Find indices where SA values exceed the threshold
    above_threshold_indices = [i for i, value in enumerate(sa_values) if not pd.isna(value) and value > threshold]
    
    if not above_threshold_indices:
        return 0.0
    
    # Initialize duration
    duration = 0.0
    # Loop through the indices to calculate the duration
    start_index = above_threshold_indices[0]
    for i in range(1, len(above_threshold_indices)):
        if above_threshold_indices[i] != above_threshold_indices[i-1] + 1:
            end_index = above_threshold_indices[i-1]
            duration += time_steps_filtered[end_index] - time_steps_filtered[start_index]
            start_index = above_threshold_indices[i]
    # Add the last interval
    end_index = above_threshold_indices[-1]
    duration += time_steps_filtered[end_index] - time_steps_filtered[start_index]
    
    return duration

# Apply the function to create the 'strong_motion_duration' column
data['strong_motion_duration'] = data.apply(calculate_strong_motion_duration, axis=1, threshold=threshold, time_steps=time_steps)

# Convert all columns to numeric, coercing errors to NaN (to remove non-numeric data)
data = data.apply(pd.to_numeric, errors='coerce')

# Handle missing values: drop rows with NaN values only in the SA columns
sa_columns = data.filter(regex=r'^SA\(').columns
data = data.dropna(subset=sa_columns)

# Keep only the SA columns and the target column
data = data[sa_columns.tolist() + ['strong_motion_duration']]

# Features and target
X = data.drop('strong_motion_duration', axis=1)
y = data['strong_motion_duration']

# Check if the dataset is empty after cleaning
if X.shape[0] == 0:
    raise ValueError("No data left after cleaning. Please check your dataset.")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Prepare new data for prediction
new_data = pd.DataFrame({
    'SA(0.010)': [-0.00363],
    'SA(0.020)': [-0.00363],
    'SA(0.030)': [-0.00363],
    'SA(0.050)': [-0.00364],
    'SA(0.075)': [-0.00365],
    'SA(0.100)': [0.00371],
    'SA(0.150)': [0.00548],
    'SA(0.200)': [0.00413],
    'SA(0.250)': [0.00431],
    'SA(0.300)': [0.00440],
    'SA(0.400)': [0.00519],
    'SA(0.500)': [0.00520],
    'SA(0.750)': [0.00793],
    'SA(1.000)': [0.00835],
    'SA(1.500)': [0.01102],
    'SA(2.000)': [0.01274],
    'SA(3.000)': [0.01023],
    'SA(4.000)': [0.00628],
    'SA(5.000)': [0.00301],
    'SA(7.500)': [0.00152],
    'SA(10.000)': [0.00083],
})

# Ensure all features are present in new data and in the same order as training data
new_data = new_data[X_train.columns]

# Predicting the duration for new data
predicted_duration = model.predict(new_data)
print(f"Predicted Strong Motion Duration: {predicted_duration[0]} seconds")
