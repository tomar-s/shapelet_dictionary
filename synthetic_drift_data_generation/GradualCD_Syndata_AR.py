import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque


def generate_ar_time_series_with_gradual_drift(n_samples=10000, window_size=20,
                                               ar_params_sets=None, drift_points=None,
                                               transition_width=500, noise_level=0.1,
                                               random_state=42):
    """
    Generate a time series dataset with autoregressive (AR) processes and gradual concept drift.

    Parameters:
    -----------
    n_samples : int
        Total number of instances to generate
    window_size : int
        Number of time steps in each sliding window
    ar_params_sets : list of lists
        List of AR parameter sets for different regimes
    drift_points : list of int
        Points where concept drift starts
    transition_width : int
        Width of the transition region where AR coefficients change gradually
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
        List of indices where concept drift starts
    """
    np.random.seed(random_state)

    # Default AR parameters if none provided
    if ar_params_sets is None:
        ar_params_sets = [
            [0.6, -0.2],  # Regime 1
            [0.9, -0.4],  # Regime 2
            [0.3, 0.6]  # Regime 3
        ]

    # Default drift points if none provided
    if drift_points is None:
        drift_points = [n_samples // 3, 2 * n_samples // 3]

    ts_length = n_samples + window_size
    time_series = np.zeros(ts_length)

    # Generate initial values
    for i in range(max(len(ar_params_sets[0]), 3)):
        time_series[i] = np.random.normal(0, 1)

    # Define regime change points
    regime_changes = [0] + drift_points + [ts_length]

    for r in range(len(regime_changes) - 1):
        start_idx = regime_changes[r]
        end_idx = regime_changes[r + 1]

        # Identify regimes for transition
        if r < len(ar_params_sets) - 1:
            regime_start = ar_params_sets[r]
            regime_end = ar_params_sets[r + 1]
        else:
            regime_start = ar_params_sets[-1]
            regime_end = ar_params_sets[-1]

        ar_order = len(regime_start)

        # Generate time series with gradual transition
        for t in range(max(start_idx, ar_order), end_idx):
            # Determine interpolation factor (0 to 1) within transition zone
            if start_idx <= t <= start_idx + transition_width:
                alpha = (t - start_idx) / transition_width  # Linear interpolation
                ar_params = [(1 - alpha) * a + alpha * b for a, b in zip(regime_start, regime_end)]
            else:
                ar_params = regime_end if t > start_idx + transition_width else regime_start

            # Compute AR component
            ar_component = sum(ar_params[i] * time_series[t - i - 1] for i in range(ar_order))

            # Add noise
            noise = np.random.normal(0, noise_level)

            # Combine
            time_series[t] = ar_component + noise

    # Create sliding window samples and assign labels
    X, y = [], []
    window = deque(maxlen=window_size)

    # Initialize window
    for i in range(window_size):
        window.append(time_series[i])

    # Generate samples
    for i in range(window_size, ts_length):
        window.append(time_series[i])

        # Determine regime label
        regime = 0
        for r, dp in enumerate(drift_points):
            if i >= dp + transition_width + window_size:
                regime = r + 1

        X.append(list(window))
        y.append(regime)

    return np.array(X), np.array(y), drift_points


# Set random seed
np.random.seed(10)


window_size = 100
n_samples = 3000
ar_params_sets = [
    [0.6, -0.2],  # Regime 0
    [0.9, -0.4],  # Regime 1
    [0.3, 0.6]  # Regime 2
]
drift_points = [2000]  # Drift starts at 1000
transition_width = 500  # Gradual transition over 500 time steps

X, y, drift_points = generate_ar_time_series_with_gradual_drift(
    n_samples=n_samples,
    window_size=window_size,
    ar_params_sets=ar_params_sets,
    drift_points=drift_points,
    transition_width=transition_width,
    noise_level=0.1
)

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f'time_step_{i}' for i in range(window_size)])
df['class'] = y
# df.to_csv('datasets/concept_drift_synthetic/gradual_using_AR_3000_seed10.csv', index=False)

# Visualize
plt.figure(figsize=(15, 8))

# Reconstruct full time series for visualization
full_ts = np.concatenate([X[0][:window_size - 1], np.array([x[-1] for x in X])])

# Plot 1: Full time series with drift points
plt.subplot(2, 1, 1)
plt.plot(full_ts)
plt.title('Time Series with Gradual Concept Drift')
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
plt.show()

print(f"Generated dataset with {len(X)} samples, each with {window_size} time steps")
print(f"Concept drift occurs at samples: {drift_points} with transition width {transition_width}")
print(f"Dataset saved to 'gradual_using_AR.csv'")
