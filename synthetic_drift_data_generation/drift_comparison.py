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


# Generate both types of drift
n_samples = 5000
drift_points = [2500]
parameter_sets = [
    [0.8, -0.2],  # First regime
    # [0.2, 0.6],  # Second regime
    [0.6, 0.3]  # Third regime
]

# Generate with abrupt drift
ts_abrupt, regimes_abrupt = generate_ar_with_abrupt_drift(
    n_samples=n_samples,
    drift_points=drift_points,
    parameter_sets=parameter_sets,
    noise_level=0.1
)

# Generate with gradual drift
ts_gradual, regimes_gradual = generate_ar_with_gradual_drift(
    n_samples=n_samples,
    drift_points=drift_points,
    drift_widths=[2000],  # Wide transition periods
    parameter_sets=parameter_sets,
    noise_level=0.1
)

# Create a unified dataset
data = pd.DataFrame({
    'time': np.arange(n_samples),
    'abrupt_drift': ts_abrupt,
    'gradual_drift': ts_gradual,
    'abrupt_regime': regimes_abrupt,
    'gradual_regime': regimes_gradual
})

# Save to CSV
data.to_csv('ar_drift_comparison.csv', index=False)

# Create visualization
plt.figure(figsize=(15, 10))

# Plot the time series
plt.subplot(3, 1, 1)
plt.plot(ts_abrupt, label='Abrupt Drift', alpha=0.8)
plt.title('AR Process with Abrupt vs Gradual Drift')
plt.ylabel('Value')
plt.legend(loc='upper right')

plt.subplot(3, 1, 2)
plt.plot(ts_gradual, label='Gradual Drift', color='green', alpha=0.8)
plt.ylabel('Value')
plt.legend(loc='upper right')

# Plot the regime labels
plt.subplot(3, 1, 3)
plt.plot(regimes_abrupt, label='Abrupt Regime Changes', drawstyle='steps-post')
plt.plot(regimes_gradual, label='Gradual Regime Changes', color='green')
plt.xlabel('Time')
plt.ylabel('Regime')
plt.legend(loc='upper right')

# Mark drift points
for dp in drift_points:
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        if i < 2:
            plt.axvline(x=dp, color='red', linestyle='--', alpha=0.5)
        else:
            plt.axvline(x=dp, color='red', linestyle='--', alpha=0.5)
            if i == 0:
                plt.text(dp, max(ts_abrupt), f'Drift at {dp}', rotation=90, verticalalignment='top')

plt.tight_layout()
plt.savefig('ar_drift_comparison.png', dpi=300)
plt.show()


# Calculate and print statistics to highlight differences
def calculate_drift_statistics(ts, drift_point, window=50):
    before = ts[drift_point - window:drift_point]
    after = ts[drift_point:drift_point + window]

    mean_change = np.mean(after) - np.mean(before)
    var_change = np.var(after) - np.var(before)

    # Calculate rate of change around drift point
    window_small = 20
    pre_drift = ts[drift_point - window_small:drift_point]
    post_drift = ts[drift_point:drift_point + window_small]

    # Simple slope estimate
    pre_slope = np.polyfit(range(window_small), pre_drift, 1)[0]
    post_slope = np.polyfit(range(window_small), post_drift, 1)[0]

    return {
        'mean_change': mean_change,
        'var_change': var_change,
        'pre_drift_slope': pre_slope,
        'post_drift_slope': post_slope,
        'slope_change': post_slope - pre_slope
    }


# Calculate statistics for both types at first drift point
abrupt_stats = calculate_drift_statistics(ts_abrupt, drift_points[0])
gradual_stats = calculate_drift_statistics(ts_gradual, drift_points[0])

print("Statistics at first drift point:")
print(f"Abrupt drift: {abrupt_stats}")
print(f"Gradual drift: {gradual_stats}")