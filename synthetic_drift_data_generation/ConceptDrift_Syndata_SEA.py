# import pandas as pd
# from scipy.io import arff
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skmultiflow.data import SEAGenerator
from collections import deque
#
# def create_univariate_sea(input_file, output_file):
#     # Read as CSV starting from the @data line
#     with open(input_file, 'r') as f:
#         content = f.read()
#
#     # Find where data starts
#     data_start = content.find('@data')
#     if data_start == -1:
#         print("Error: @data section not found")
#         return
#
#     # Extract header information and data section
#     header = content[:data_start]
#     data_section = content[data_start + 5:].strip()
#
#     # Parse data rows
#     rows = []
#     for line in data_section.split('\n'):
#         if line.strip() and not line.startswith('%'):
#             # Split by comma and remove trailing comma if exists
#             values = line.rstrip(',').split(',')
#             if len(values) >= 2:
#                 # Keep only first attribute and class
#                 rows.append([values[0], values[-1]])
#
#     # Write new ARFF file
#     with open(output_file, 'w') as f:
#         f.write("@relation 'UnivariateSEA'\n")
#         f.write("@attribute feature1 numeric\n")
#         f.write("@attribute class {groupA,groupB}\n")
#         f.write("@data\n")
#
#         for row in rows:
#             f.write(f"{row[0]},{row[1]}\n")
#
#     print(f"Created univariate dataset with {len(rows)} instances in {output_file}")
#
# # Use the function
# create_univariate_sea('/Users/shivanitomar/moa_data_1/gradual_drift_3.arff', '/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/datasets/concept_drift_synthetic/univariate_sea_gradual_3.arff')
#
# g_data, meta = arff.loadarff("/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/datasets/concept_drift_synthetic/univariate_sea_gradual.arff")
# g_data_df = pd.DataFrame(g_data)
#
# print("dataframe", g_data_df)


def generate_time_series_with_drift(n_samples=10000, window_size=10, n_drift_points=3, random_state=42):
    """
    Generate a univariate time series dataset with concept drift using the SEA Generator.

    Parameters:
    -----------
    n_samples : int
        Total number of instances to generate
    window_size : int
        Number of time steps in each sliding window that forms a single instance
    n_drift_points : int
        Number of concept drift points to introduce
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X : numpy.ndarray
        Array of shape (n_samples, window_size) where each row is a time window
    y : numpy.ndarray
        Array of shape (n_samples,) containing class labels
    drift_points : list
        List of indices where concept drift occurs
    """
    # Calculate where drift will occur
    drift_points = [int(n_samples * (i + 1) / (n_drift_points + 1)) for i in range(n_drift_points)]

    # Lists to store the generated data
    all_values = []
    all_labels = []

    # Define a sliding window to create time series instances
    window = deque(maxlen=window_size)

    # Initialize with a random function (0, 1, or 2)
    current_function = 0

    # Initialize the first SEA generator with function 0
    stream = SEAGenerator(classification_function=current_function, random_state=random_state)

    # Fill the initial window
    for _ in range(window_size - 1):
        X, y = stream.next_sample()
        window.append(X[0][0])  # Take just the first feature

    # Generate the main data
    for i in range(n_samples):
        # Introduce concept drift at specified points
        if i in drift_points:
            # Change to a different classification function (cycle through 0, 1, 2)
            current_function = (current_function + 1) % 3

            # Create a new generator with the new function
            stream = SEAGenerator(classification_function=current_function, random_state=random_state + i)

        # Get next sample
        X, y = stream.next_sample()

        # Add to window
        window.append(X[0][0])  # Using only the first feature

        # Store the current window as a sample
        all_values.append(list(window))
        all_labels.append(y[0])

    return np.array(all_values), np.array(all_labels), drift_points


# Example usage
X, y, drift_points = generate_time_series_with_drift(n_samples=2000, window_size=100, n_drift_points=1)

# Convert to DataFrame for easier handling
df = pd.DataFrame(X, columns=[f'time_step_{i}' for i in range(X.shape[1])])
df['class'] = y

# Save to CSV
# df.to_csv('sea_time_series_with_drift.csv', index=False)

# Visualize the data and drift points
plt.figure(figsize=(15, 6))

# Plot the class distribution over time
plt.subplot(2, 1, 1)
plt.scatter(range(len(y)), y, s=5, alpha=0.5)
plt.title('Class Distribution Over Time')
plt.ylabel('Class')

# Mark drift points
for dp in drift_points:
    plt.axvline(x=dp, color='red', linestyle='--', alpha=0.7)
    plt.text(dp, 0.5, f'Drift at {dp}', rotation=90, verticalalignment='center')

# Plot time series samples at different points
plt.subplot(2, 1, 2)
sample_indices = [500, 1500]  # Before, between and after drift points
for idx in sample_indices:
    plt.plot(X[idx], label=f'Sample at {idx}')

plt.title('Example Time Series Windows')
plt.xlabel('Time Step Within Window')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
# plt.savefig('time_series_drift_visualization.png')
plt.show()

print(f"Generated dataset with {len(X)} samples, each with {X.shape[1]} time steps")
print(f"Concept drift occurs at samples: {drift_points}")
print(f"Dataset saved to 'sea_time_series_with_drift.csv'")






# # Load the ARFF file
# data, meta = arff.loadarff("/Users/shivanitomar/Downloads/electricity-normalized (1).arff")
# df = pd.DataFrame(data)
