import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_ar_with_gradual_drift(n_samples=5000, ar_order=2, drift_points=None,
                                   drift_widths=None, parameter_sets=None,
                                   noise_level=0.1, random_state=42):
    """
    Generate AR process with gradual concept drift.

    Parameters:
    -----------
    n_samples : int
        Total length of the time series
    ar_order : int
        Order of the AR process
    drift_points : list
        Center points where drift occurs
    drift_widths : list
        Width of each drift transition period
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
        Labels indicating which regime generated each point (fractional during transitions)
    """
    np.random.seed(random_state)

    # Default drift points if none provided
    if drift_points is None:
        drift_points = [n_samples // 3, 2 * n_samples // 3]

    # Default drift widths if none provided
    if drift_widths is None:
        drift_widths = [500, 500]  # Width of transition periods

    # Default parameter sets if none provided
    if parameter_sets is None:
        parameter_sets = [
            [0.6, -0.2],  # First regime
            [0.9, -0.5],  # Second regime
            [0.3, 0.4]  # Third regime
        ]

    # Initialize time series and regime labels
    ts = np.zeros(n_samples)
    regime_labels = np.zeros(n_samples)

    # Initial values
    for i in range(ar_order):
        ts[i] = np.random.normal(0, 1)

    # Calculate stable regime periods and transition periods
    transitions = []
    stable_periods = []

    current_pos = 0
    for i, dp in enumerate(drift_points):
        width = drift_widths[i]

        # Start of transition
        transition_start = max(dp - width // 2, current_pos)

        # Add stable period before transition
        if transition_start > current_pos:
            stable_periods.append((current_pos, transition_start, i))

        # Add transition period
        transition_end = min(dp + width // 2, n_samples)
        transitions.append((transition_start, transition_end, i, i + 1))

        current_pos = transition_end

    # Add final stable period
    if current_pos < n_samples:
        stable_periods.append((current_pos, n_samples, len(parameter_sets) - 1))

    # Generate stable periods
    for start, end, regime in stable_periods:
        params = parameter_sets[regime]

        for t in range(max(start, ar_order), end):
            # Calculate AR component
            ar_component = sum(params[i] * ts[t - i - 1] for i in range(min(ar_order, len(params))))

            # Add noise
            noise = np.random.normal(0, noise_level)

            # Set value and label
            ts[t] = ar_component + noise
            regime_labels[t] = regime

    # Generate transition periods
    for start, end, regime1, regime2 in transitions:
        params1 = parameter_sets[regime1]
        params2 = parameter_sets[regime2]

        for t in range(max(start, ar_order), end):
            # Calculate transition progress (0 to 1)
            progress = (t - start) / (end - start)

            # Interpolate parameters
            current_params = []
            for i in range(max(len(params1), len(params2))):
                p1 = params1[i] if i < len(params1) else 0
                p2 = params2[i] if i < len(params2) else 0
                current_params.append(p1 * (1 - progress) + p2 * progress)

            # Calculate AR component
            ar_component = sum(current_params[i] * ts[t - i - 1] for i in range(min(ar_order, len(current_params))))

            # Add noise
            noise = np.random.normal(0, noise_level)

            # Set value and label
            ts[t] = ar_component + noise
            regime_labels[t] = regime1 * (1 - progress) + regime2 * progress

    return ts, regime_labels


# Example usage
n_samples = 5000
drift_points = [1500, 3500]
drift_widths = [800, 800]  # Wide transition periods for gradual drift
parameter_sets = [
    [0.8, -0.2],  # First regime
    [0.2, 0.6],  # Second regime
    [0.6, 0.3]  # Third regime
]

# Generate time series with gradual drift
ts_gradual, regimes_gradual = generate_ar_with_gradual_drift(
    n_samples=n_samples,
    drift_points=drift_points,
    drift_widths=drift_widths,
    parameter_sets=parameter_sets,
    noise_level=0.1
)

# Plot results
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.plot(ts_gradual)
plt.title('AR Process with Gradual Drift')
plt.xlabel('Time')
plt.ylabel('Value')

for dp in drift_points:
    plt.axvline(x=dp, color='red', linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(regimes_gradual)
plt.title('Regime Progression (Shows Gradual Transitions)')
plt.xlabel('Time')
plt.ylabel('Regime')

plt.tight_layout()
plt.show()