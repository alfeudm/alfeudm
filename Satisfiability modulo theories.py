import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize

# Define the sequence length and the range of values
sequence_length = 5
value_range = range(1, 80)

# Define the number of predicted sequences
num_predictions = 10

# Path to the CSV file
csv_file = "quina5de80.csv"

# Read observed sequences from the CSV file
observed_sequences = []
with open(csv_file, "r") as file:
    reader = csv.reader(file, delimiter=";")
    next(reader)  # Skip the header row
    for row in reader:
        # Skip blank lines
        if not any(row):
            continue
        observed_sequence = [int(cell) for cell in row]
        observed_sequences.append(observed_sequence)

# Train a machine learning model for sequence prediction
X_train = np.array(observed_sequences[:-1])  # Use all but the last observed sequence as training data
y_train = np.array(observed_sequences[1:])   # Predict the next sequence
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

# Define the objective function
def objective_function(predictions):
    total_distance = 0
    for observed_sequence in observed_sequences:
        min_distance = float("inf")
        for predicted_sequence in predictions:
            distance = np.sum(np.abs(np.array(observed_sequence) - np.array(predicted_sequence)))
            min_distance = min(min_distance, distance)
        total_distance += min_distance
    return total_distance

# Define the constraints
def constraints(predictions):
    constr = []
    if predictions.ndim == 1:
        predictions = [predictions]
    for prediction in predictions:
        prediction_list = prediction.tolist()  # Convert to list
        for value in prediction_list:
            constr.append({'type': 'ineq', 'fun': lambda x: value - x})
            constr.append({'type': 'ineq', 'fun': lambda x: x - value - 1})
    return constr

# Generate initial predictions using the machine learning model
initial_predictions = []
last_observed_sequence = observed_sequences[-1]  # Last observed sequence as input
for _ in range(num_predictions):
    prediction = regressor.predict([last_observed_sequence])
    initial_predictions.append(prediction.flatten())
    last_observed_sequence = prediction.flatten()

# Reshape initial predictions to 1D
initial_predictions = np.concatenate(initial_predictions)

# Adjust the number of predictions based on available data
num_predictions = min(num_predictions, len(observed_sequences) - 1)

# Perform the optimization
result = minimize(objective_function, initial_predictions.ravel(), method='SLSQP', constraints=constraints(initial_predictions))
optimized_predictions = result.x

# Reshape the optimized predictions
optimized_predictions = optimized_predictions.reshape(num_predictions, sequence_length)

# Print the predicted sequences
print("Predicted Sequences:")
for predicted_sequence in optimized_predictions:
    print(predicted_sequence)
