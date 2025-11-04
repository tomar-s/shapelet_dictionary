import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_ar_with_abrupt_drift(n_samples=5000, ar_order=2, drift_points=None,
                                  parameter_sets=None, noise_level=0.1, random_state=42):
    """
    Generate AR process with abrupt concept drift.

    Parameters:
    -----------
    n_samples : int
        Total length of the time series
    ar_order : int
        Order of the AR process
    drift_points : list
        Points where drift occurs
    parameter_sets : list of lists
        List of AR parameter sets for each regime
    noise_level : float
        Standard deviation of noise
    random_state : int
        Random seed

    Returns:
    --------
    ts : numpy.ndarray
        Generated time series
    regime_labels : numpy.ndarray
        Labels indicating which regime generated each point
    """
    np.random.seed(random_state)

    # Default drift points if none provided
    if drift_points is None:
        drift_points = [n_samples // 3, 2 * n_samples // 3]

    # Default parameter sets if none provided
    if parameter_sets is None:
        parameter_sets = [
            [0.6, -0.2],  # First regime
            [0.9, -0.5],  # Second regime
            [0.3, 0.4]  # Third regime
        ]

    # Initialize time series and regime labels
    ts = np.zeros(n_samples)
    regime_labels = np.zeros(n_samples, dtype=int)

    # Initial values
    for i in range(ar_order):
        ts[i] = np.random.normal(0, 1)

    # Add drift points to regime boundaries
    regime_boundaries = [0] + drift_points + [n_samples]

    # Generate time series with abrupt drift
    for r in range(len(regime_boundaries) - 1):
        start_idx = regime_boundaries[r]
        end_idx = regime_boundaries[r + 1]

        # Current parameters
        params = parameter_sets[r]

        # Generate values for current regime
        for t in range(max(start_idx, ar_order), end_idx):
            # Calculate AR component
            ar_component = sum(params[i] * ts[t - i - 1] for i in range(min(ar_order, len(params))))

            # Add noise
            noise = np.random.normal(0, noise_level)

            # Set value
            ts[t] = ar_component + noise

            # Set regime label
            regime_labels[t] = r

    return ts, regime_labels


# Example usage
n_samples = 5000
drift_points = [2500]
parameter_sets = [
    [0.8, -0.2],  # First regime
    # [0.2, 0.6],  # Second regime
    [0.6, 0.3]  # Third regime
]

# Generate time series with abrupt drift
ts_abrupt, regimes_abrupt = generate_ar_with_abrupt_drift(
    n_samples=n_samples,
    drift_points=drift_points,
    parameter_sets=parameter_sets,
    noise_level=0.1
)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(ts_abrupt)
plt.title('AR Process with Abrupt Drift')
plt.xlabel('Time')
plt.ylabel('Value')

for dp in drift_points:
    plt.axvline(x=dp, color='red', linestyle='--')
    plt.text(dp, min(ts_abrupt), f'Drift at {dp}', rotation=90)

plt.tight_layout()
plt.show()