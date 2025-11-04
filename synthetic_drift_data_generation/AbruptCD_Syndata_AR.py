import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque


def generate_ar_time_series_with_drift(n_samples=10000, window_size=20,
                                       ar_params_sets=None, drift_points=None,
                                       noise_level=0.1, random_state=42):
    """
    Generate a time series dataset based on autoregressive processes with concept drift.

    Parameters:
    -----------
    n_samples : int
        Total number of instances to generate
    window_size : int
        Number of time steps in each sliding window that forms a single instance
    ar_params_sets : list of lists
        List of AR parameter sets, each defining a different regime
    drift_points : list of int
        Points where concept drift occurs
    noise_level : float
        Standard deviation of Gaussian noise added to the series
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X : numpy.ndarray
        Array of shape (n_samples, window_size) with time series windows
    y : numpy.ndarray
        Array of shape (n_samples,) with class labels
    drift_points : list
        List of indices where concept drift occurs
    """
    np.random.seed(random_state)

    # Default AR parameters if none provided (3 different regimes)
    if ar_params_sets is None:
        ar_params_sets = [
            [0.6, -0.2, 0.1],  # Regime 1: moderate oscillation
            [0.9, -0.5, 0.2],  # Regime 2: stronger oscillation
            [0.3, 0.3, -0.1]  # Regime 3: different pattern
        ]

    # Default drift points if none provided
    if drift_points is None:
        drift_points = [n_samples // 3, 2 * n_samples // 3]

    # Initialize the time series
    ts_length = n_samples + window_size  # Extra points to initialize the window
    time_series = np.zeros(ts_length)

    # Generate initial values
    for i in range(max(len(ar_params_sets[0]), 3)):
        time_series[i] = np.random.normal(0, 1)

    # Generate the complete time series with drift
    current_regime = 0
    regime_changes = [0] + drift_points + [ts_length]

    for r in range(len(regime_changes) - 1):
        start_idx = regime_changes[r]
        end_idx = regime_changes[r + 1]

        if r < len(ar_params_sets):
            current_regime = r

        ar_params = ar_params_sets[current_regime]
        ar_order = len(ar_params)

        # Generate time series for this regime
        for t in range(max(start_idx, ar_order), end_idx):
            # AR component
            ar_component = sum(ar_params[i] * time_series[t - i - 1] for i in range(ar_order))

            # Add noise
            noise = np.random.normal(0, noise_level)

            # Combine
            time_series[t] = ar_component + noise

    # Create sliding window samples and assign labels
    X = []
    y = []
    window = deque(maxlen=window_size)

    # Initialize window
    for i in range(window_size):
        window.append(time_series[i])

    # Generate samples
    for i in range(window_size, ts_length):
        # Add new value to window
        window.append(time_series[i])

        # Determine which regime this sample belongs to
        regime = 0
        for r, dp in enumerate(drift_points):
            if i >= dp + window_size:
                regime = r + 1

        # Store window and label
        X.append(list(window))
        y.append(regime)  # Use regime number as class label

    return np.array(X), np.array(y), drift_points


# Example usage
window_size = 100
n_samples = 2000
ar_params_sets = [
    [0.6, -0.2],  # Regime 0
    # [0.9, -0.4],  # Regime 1
    [0.3, 0.6]  # Regime 2
]
drift_points = [1000]  # Points where concept drift occurs

X, y, drift_points = generate_ar_time_series_with_drift(
    n_samples=n_samples,
    window_size=window_size,
    ar_params_sets=ar_params_sets,
    drift_points=drift_points,
    noise_level=0.1
)

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f'time_step_{i}' for i in range(window_size)])
df['class'] = y
# df.to_csv('datasets/concept_drift_synthetic/gradual_using_AR.csv', index=False)

# Visualize
plt.figure(figsize=(15, 8))

# Reconstruct the full time series for visualization
full_ts = np.concatenate([X[0][:window_size - 1], np.array([x[-1] for x in X])])

# Plot 1: Full time series with drift points
plt.subplot(2, 1, 1)
plt.plot(full_ts)
plt.title('Time Series with Abrupt Concept Drift')
plt.ylabel('Value')

for dp in drift_points:
    plt.axvline(x=dp, color='red', linestyle='--', alpha=0.7)
    plt.text(dp, min(full_ts), f'Drift at {dp}', rotation=90, verticalalignment='bottom')

# Plot 2: Class distribution
plt.subplot(2, 1, 2)
plt.scatter(range(len(y)), y, s=2)
plt.title('Class Distribution (Regimes)')
plt.xlabel('Time')
plt.ylabel('Regime')
plt.yticks(np.unique(y))

plt.tight_layout()
# plt.savefig('ar_time_series_drift_visualization.png')
plt.show()

print(f"Generated dataset with {len(X)} samples, each with {window_size} time steps")
print(f"Concept drift occurs at samples: {drift_points}")
print(f"Dataset saved to 'ar_time_series_with_drift.csv'")