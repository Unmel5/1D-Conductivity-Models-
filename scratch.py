import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import tensorflow as tf

# Define the grid parameters.
total_points = 51
grid_spacing = 4.0

# Create a grid of points.
grid_points = np.linspace(0.0, total_points * grid_spacing, total_points)

# Print the grid points.
print(grid_points)

depth_range = 200
num_resistivities = random.randint(1, 15)
#generates a list of num_resisitivities random numbers between 0 and depth_range , the list is then sorted in ascending order using the sorted() function
dominant_depths = sorted([random.uniform(0, depth_range) for _ in range(num_resistivities)])
mean_resistivity = 1
std_deviation = 1.5
resistivity_values = [10 ** np.random.normal(mean_resistivity, std_deviation) for _ in range(num_resistivities)]
resistivity_values = [max(0.1, min(10000, value)) for value in resistivity_values]
for i, (depth, resistivity) in enumerate(zip(dominant_depths, resistivity_values), start=1):
      print(f"Resistivity {i}: Depth = {depth:.2f} meters, Resistivity = {resistivity:.2f} ohm·m")

# Function to generate a single resistivity model
def generate_resistivity_model(depth_range, mean_resistivity, std_deviation):
    num_resistivities = random.randint(1, 15)
    dominant_depths = sorted([random.uniform(0, depth_range) for _ in range(num_resistivities)])
    resistivity_values = [10 ** np.random.normal(mean_resistivity, std_deviation) for _ in range(num_resistivities)]
    resistivity_values = [max(0.1, min(10000, value)) for value in resistivity_values]
    return dominant_depths, resistivity_values

# Parameters for the 51-point vertical grid
grid_points = np.linspace(0.0, 200.0, 51)

models = [] #define the models
accepted_models = []
interpolated_models = []

for i in range(100000): #generate with a for loop
    depths, resistivities = generate_resistivity_model(depth_range=200, mean_resistivity=1, std_deviation=1.5)

    # one error I kept getting was that there needed to be two depths for spline. "ValueError: `x` must contain at least 2 elements." this code makes sure it has 2 or greater.
    if len(depths) >= 2:
        # Check if all resistivities are within the range before adding to the accepted models
        if all(0.1 <= r <= 10000 for r in resistivities): #"Some of them will be rejected because their resistivity values are outside the range [0.1,10000]"
            accepted_models.append((depths, resistivities)) #adding a tuple containing "depths" and "resisitivities" to the list of "accepted models"

            spline = CubicSpline(depths, resistivities)
            interpolated_resistivity = spline(grid_points)

            # Ensure interpolated resistivities are also within the 100000 range.
            if all(0.1 <= r <= 10000 for r in interpolated_resistivity):
                interpolated_models.append(interpolated_resistivity)



# Prepare data for the neural network
em_responses = np.array(interpolated_models)
train_split = 0.8
train_size = int(train_split * len(em_responses))
train_idx, val_idx = np.split(np.arange(len(em_responses)), [train_size])
train_data = np.random.rand(len(em_responses), grid_points.shape[0])  # Dummy train_data

# Neural Network setup
input_shape = em_responses.shape[1]
output_neurons = train_data.shape[1]
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_shape,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(output_neurons)
])

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
history = model.fit(em_responses[train_idx], train_data[train_idx],
                    validation_data=(em_responses[val_idx], train_data[val_idx]),
                    epochs=100, batch_size=32)

# Evaluate and visualize the model's performance
val_loss = model.evaluate(em_responses[val_idx], train_data[val_idx])
print("Validation loss:", val_loss)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

# Predicted vs Actual Resistivity Profile Visualization
sample_index = np.random.randint(0, len(val_idx))
predicted_profile = model.predict(em_responses[val_idx][sample_index].reshape(1, -1)).flatten()
plt.plot(predicted_profile, grid_points, label='Predicted')
plt.plot(train_data[val_idx][sample_index], grid_points, label='Actual')
plt.gca().invert_yaxis()
plt.xlabel('Resistivity (Ohm-m)')
plt.ylabel('Depth (m)')
plt.title('Predicted vs Actual Resistivity')
plt.legend()
plt.show()

