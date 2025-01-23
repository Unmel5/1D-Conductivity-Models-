Read-Me: Resistivity Profile Modeling and Neural Network Prediction

Overview

This script generates synthetic resistivity models based on a vertical grid and uses these models to train a neural network for resistivity profile predictions. The script includes steps for data generation, interpolation, and training a neural network. Visualizations are also provided to evaluate the model's performance.

Key Features

Resistivity Model Generation:

Randomly generates resistivity models with varying depths and values.

Ensures resistivity values are within a valid range [0.1, 10000] ohm·m.

Interpolation:

Uses cubic splines to interpolate resistivity values onto a 51-point vertical grid.

Data Preparation:

Generates synthetic electromagnetic (EM) responses as input data for the neural network.

Splits the dataset into training and validation sets.

Neural Network Setup:

A dense neural network with two hidden layers is used.

The model is trained using the mean squared error (MSE) loss function and Adam optimizer.

Visualization:

Plots training and validation losses.

Compares predicted and actual resistivity profiles for sample data.

Code Components

1. Grid Definition

A 51-point vertical grid is defined with a spacing of 4 meters:# Define the grid parameters.
total_points = 51
grid_spacing = 4.0

# Create a grid of points.
grid_points = np.linspace(0.0, total_points * grid_spacing, total_points)

# Print the grid points.
print(grid_points) 2. Resistivity Model Generation

Dominant depths and resistivity values are generated randomly.

Resistivity values follow a log-normal distribution with specified mean and standard deviation. 

Example code:
# Function to generate a single resistivity model
def generate_resistivity_model(depth_range, mean_resistivity, std_deviation):
    num_resistivities = random.randint(1, 15)
    dominant_depths = sorted([random.uniform(0, depth_range) for _ in range(num_resistivities)])
    resistivity_values = [10 ** np.random.normal(mean_resistivity, std_deviation) for _ in range(num_resistivities)]
    resistivity_values = [max(0.1, min(10000, value)) for value in resistivity_values]
    return dominant_depths, resistivity_values



# Function to generate a single resistivity model
def generate_resistivity_model(depth_range, mean_resistivity, std_deviation):
    num_resistivities = random.randint(1, 15)
    dominant_depths = sorted([random.uniform(0, depth_range) for _ in range(num_resistivities)])
    resistivity_values = [10 ** np.random.normal(mean_resistivity, std_deviation) for _ in range(num_resistivities)]
    resistivity_values = [max(0.1, min(10000, value)) for value in resistivity_values]
    return dominant_depths, resistivity_values

3. Cubic Spline Interpolation

Interpolates resistivity values onto the grid, ensuring all interpolated values remain within the valid range:

# Generate and interpolate resistivity models
models = [] # define the models
accepted_models = []
interpolated_models = []

for i in range(100000):
    depths, resistivities = generate_resistivity_model(depth_range=200, mean_resistivity=1, std_deviation=1.5)

    if len(depths) >= 2: # Ensure spline interpolation can be performed
        if all(0.1 <= r <= 10000 for r in resistivities):
            accepted_models.append((depths, resistivities))

            spline = CubicSpline(depths, resistivities)
            interpolated_resistivity = spline(grid_points)

            if all(0.1 <= r <= 10000 for r in interpolated_resistivity):
                interpolated_models.append(interpolated_resistivity

4. Neural Network Training

Input: Interpolated resistivity models.

Output: Synthetic training data (dummy data for demonstration).

Neural network setup:
from tensorflow.keras import layers

em_responses = np.array(interpolated_models)
train_split = 0.8
train_size = int(train_split * len(em_responses))
train_idx, val_idx = np.split(np.arange(len(em_responses)), [train_size])
train_data = np.random.rand(len(em_responses), grid_points.shape[0])  # Dummy train_data

input_shape = em_responses.shape[1]
output_neurons = train_data.shape[1]
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_shape,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(output_neurons)
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(em_responses[train_idx], train_data[train_idx],
                    validation_data=(em_responses[val_idx], train_data[val_idx]),
                    epochs=100, batch_size=32)

4. Neural Network Training

Input: Interpolated resistivity models.

Output: Synthetic training data (dummy data for demonstration).
from tensorflow.keras import layers

em_responses = np.array(interpolated_models)
train_split = 0.8
train_size = int(train_split * len(em_responses))
train_idx, val_idx = np.split(np.arange(len(em_responses)), [train_size])
train_data = np.random.rand(len(em_responses), grid_points.shape[0])  # Dummy train_data

input_shape = em_responses.shape[1]
output_neurons = train_data.shape[1]
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_shape,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(output_neurons)
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(em_responses[train_idx], train_data[train_idx],
                    validation_data=(em_responses[val_idx], train_data[val_idx]),
                    epochs=100, batch_size=32)

5. Model Evaluation and Visualization

Training and validation losses are plotted to assess model convergence.

A sample predicted resistivity profile is compared to the actual profile for qualitative evaluation:

val_loss = model.evaluate(em_responses[val_idx], train_data[val_idx])
print("Validation loss:", val_loss)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

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

Instructions for Use

Dependencies: Install the following Python libraries:

numpy

matplotlib

scipy

tensorflow

Install via pip if not already available:

pip install numpy matplotlib scipy tensorflow

Run the Script: Execute the script in your Python environment. The script will:

Generate resistivity models.

Train a neural network.

Visualize the results.

Output:

Training and validation loss plot.

Visualization of a predicted resistivity profile against the actual profile.

Solution to GitHub Issue

To resolve the GitHub branch issue:

Check your branch:

git branch

If on master, rename it to main:

git branch -M main

Add and commit changes:

git add scratch.py
git commit -m "Add scratch.py file for resistivity model and NN"

Push to remote repository:

git push -u origin main

If the error persists, ensure the remote URL is correct:

git remote set-url origin https://github.com/Unmel5/1D-Conductivity-Models-.git

Notes

The training data in this script is generated as dummy data for demonstration purposes.

For real-world applications, replace the dummy data with actual field measurements or simulated EM responses.

Ensure the generated resistivity values and interpolated models satisfy the required physical constraints.




