import numpy as np
import pandas as pd
import time
import heapq
from scipy.spatial.distance import euclidean
from scipy.io import loadmat, savemat, arff
import matplotlib.pyplot as plt



def USIDL(X, y, lambda_, K, q, c, epsilon, maxIter, maxInnerIter, runid):
    """
    X: n x p, training data, n times series with length p
    y: binary label (-1, 1), not used in this model, just for plotting
    lambda_: regularization parameter for l1 norm
    K: number of basis
    q: length of basis over
    c: Squared L2-norm of basis, i.e., ||s_k||^2 <= c
    epsilon: epsilon
    maxIter: maximum outer iterations
    maxInnerIter: maximum inner iterations
    runid: magic string prefix for plotting

    Returns:
        S: learned basis
        A: coefficients for training data
        Offsets: matched location of the basis
        F_obj: array of objective values
    """
    n, p = X.shape

    S = np.random.randn(K, q)  # initialize bases
    A = np.random.randn(n, K)  # basis initializations
    Offsets = np.random.randint(0, p - q + 1, (n, K))  # initialize offsets

    F_obj = []

    for iter in range(int(maxIter)):
        # update coefficients and matching offsets
        A, Offsets, F_all = update_A_par(X, S, A, Offsets, lambda_, maxInnerIter, epsilon)

        # update bases
        S = update_S(X, S, A, Offsets, lambda_, c, maxInnerIter, epsilon)

        # check convergence
        F_all = unsup_obj(X, S, A, Offsets, lambda_)

        F_obj.append(F_all)
        if len(F_obj) > 1 and abs(F_obj[-1] - F_obj[-2]) / F_obj[-2] < epsilon:
            print('Converged!')
            return S, A, Offsets, F_obj

    print('Maximum Iteration Reached!')
    return S, A, Offsets, F_obj

def USIDL_with_alpha_const(X, y, lambda_, K, q, c, epsilon, maxIter, maxInnerIter, runid):
    """
    X: n x p, training data, n times series with length p
    y: binary label (-1, 1), not used in this model, just for plotting
    lambda_: regularization parameter for l1 norm
    K: number of basis
    q: length of basis over
    c: Squared L2-norm of basis, i.e., ||s_k||^2 <= c
    epsilon: epsilon
    maxIter: maximum outer iterations
    maxInnerIter: maximum inner iterations
    runid: magic string prefix for plotting

    Returns:
        S: learned basis
        A: coefficients for training data
        Offsets: matched location of the basis
        F_obj: array of objective values
    """
    n, p = X.shape

    # S = np.random.randn(K, q)  # randomly initialize bases
    # S = init_dictionary_with_svd(X, K , q)
    S = init_dictionary_with_random_chunks(X, K , q) #initializing dictionary from random chunks of data
    A = np.random.rand(n, K)  # initializations of alpha change to allow only positive values
    Offsets = np.random.randint(0, p - q + 1, (n, K))  # initialize offsets

    F_obj = []

    for iter in range(int(maxIter)):
        # update coefficients and matching offsets
        A, Offsets, F_all = update_A_par_with_alpha_const(X, S, A, Offsets, lambda_, maxInnerIter, epsilon)

        # update bases
        S = update_S(X, S, A, Offsets, lambda_, c, maxInnerIter, epsilon)

        # check convergence
        F_all = unsup_obj(X, S, A, Offsets, lambda_)


        F_obj.append(F_all)
        if len(F_obj) > 1 and abs(F_obj[-1] - F_obj[-2]) / F_obj[-2] < epsilon:
            print('Converged!')
            return S, A, Offsets, F_obj

    print('Maximum Iteration Reached!')
    return S, A, Offsets, F_obj

def update_A_par_with_alpha_const(X, S, A, Offsets, lambda_, maxIter, epsilon):
    """
    X: n x p
    S: K x q
    A: n x K
    Offsets: n x K

    Returns:
        A: updated coefficients for training data
        Offsets: updated matched location of the basis
        F_all: final objective value
    """
    n, p = X.shape
    KK, q = S.shape
    seg_idx = np.add.outer(np.arange(p - q + 1), np.arange(q))

    F_obj = []
    for iter in range(int(maxIter)):
        for i in range(n):  # compute activation and matching offset for X_i
            x = X[i, :]
            offs = Offsets[i, :]
            shifted_S = op_shift(S, offs, p)

            # compute for base k
            for k in np.random.permutation(KK):
                base = S[k, :]
                temp_a = A[i, :].copy()
                temp_a[k] = 0  # exclude alpha_k

                x_residue = x - np.dot(temp_a, shifted_S)
                residue_norm2 = np.linalg.norm(x_residue) ** 2
                base_norm2 = np.linalg.norm(base) ** 2  # ||s_k||^2

                segs = x_residue[seg_idx]
                dot_prods = np.dot(segs, base)

                M_idx = np.argmax(np.abs(dot_prods))
                M_dp = dot_prods[M_idx]

                if M_dp <= lambda_:
                    a_k_star = 0
                else:
                    a_k_star = (M_dp - lambda_) / base_norm2
                    t_k_star = M_idx

                A[i, k] = a_k_star
                if a_k_star != 0:
                    shifted_S[k, :] = 0
                    shifted_S[k, t_k_star: t_k_star + q] = base
                    Offsets[i, k] = t_k_star

        F_all = unsup_obj(X, S, A, Offsets, lambda_)
        F_obj.append(F_all)
        if len(F_obj) > 1 and abs(F_obj[-1] - F_obj[-2]) / F_obj[-2] < epsilon:
            # print('Updating A: Converged!\n\n')
            return A, Offsets, F_all

    # print('Updating A: Reached max iter.\n\n')
    return A, Offsets, F_all


def op_shift(S, offsets, target_dim):
    """
    Shifts the basis functions according to the given offsets.

    Parameters:
    S (numpy.ndarray): Basis functions (K x q)
    offsets (numpy.ndarray): Offsets (1 x K)
    target_dim (int): Target dimension for the shifted basis functions

    Returns:
    numpy.ndarray: Shifted basis functions (K x target_dim)
    """
    K, q = S.shape

    res = np.zeros((K, target_dim))
    for i in range(K):
        offset = offsets[i]
        if offset + q <= target_dim:
            res[i, offset:offset + q] = S[i]
        else:
            raise ValueError(f"Offset {offset} with q {q} exceeds target_dim {target_dim}")

    return res

def update_A_par(X, S, A, Offsets, lambda_, maxIter, epsilon):
    """
    X: n x p
    S: K x q
    A: n x K
    Offsets: n x K

    Returns:
        A: updated coefficients for training data
        Offsets: updated matched location of the basis
        F_all: final objective value
    """
    n, p = X.shape
    KK, q = S.shape
    seg_idx = np.add.outer(np.arange(p - q + 1), np.arange(q))

    F_obj = []
    for iter in range(int(maxIter)):
        for i in range(n):  # compute activation and matching offset for X_i
            x = X[i, :]
            offs = Offsets[i, :]
            shifted_S = op_shift(S, offs, p)

            # compute for base k
            for k in np.random.permutation(KK):
                base = S[k, :]
                temp_a = A[i, :].copy()
                temp_a[k] = 0  # exclude alpha_k

                x_residue = x - np.dot(temp_a, shifted_S)
                residue_norm2 = np.linalg.norm(x_residue) ** 2
                base_norm2 = np.linalg.norm(base) ** 2  # ||s_k||^2

                segs = x_residue[seg_idx]
                dot_prods = np.dot(segs, base)

                M_idx = np.argmax(np.abs(dot_prods))
                M_dp = dot_prods[M_idx]

                if np.abs(M_dp) <= lambda_:
                    a_k_star = 0
                else:
                    a_k_star = np.sign(M_dp) * (np.abs(M_dp) - lambda_) / base_norm2
                    t_k_star = M_idx

                A[i, k] = a_k_star
                if a_k_star != 0:
                    shifted_S[k, :] = 0
                    shifted_S[k, t_k_star: t_k_star + q] = base
                    Offsets[i, k] = t_k_star

        F_all = unsup_obj(X, S, A, Offsets, lambda_)
        F_obj.append(F_all)
        if len(F_obj) > 1 and abs(F_obj[-1] - F_obj[-2]) / F_obj[-2] < epsilon:
            # print('Updating A: Converged!\n\n')
            return A, Offsets, F_all

    # print('Updating A: Reached max iter.\n\n')
    return A, Offsets, F_all


def update_S(X, S, A, Offsets, lambda_, c, maxIter, epsilon):
    """
    X: n x p
    S: K x q
    A: n x K
    Offsets: n x K

    Returns:
        S: updated basis functions
    """
    n, p = X.shape
    K, q = S.shape

    F_obj = []

    for iter in range(int(maxIter)):
        for k in range(K):  # optimize s_k
            M_k = np.linalg.norm(A[:, k]) ** 2
            if M_k == 0:  # inactive bases, no need to update
                continue

            # s_k = np.zeros(q)
            s_k = S[k]

            for i in range(n):
                temp_a = A[i, :].copy()
                temp_a[k] = 0
                shifted_S = op_shift(S, Offsets[i, :], p)
                xi_residue = X[i, :] - np.dot(temp_a, shifted_S)

                t_ik = Offsets[i, k]
                s_k += A[i, k] * xi_residue[t_ik:t_ik + q]

            # compute s_k
            if M_k <= np.linalg.norm(s_k) / np.sqrt(c):
                s_k = np.sqrt(c) / np.linalg.norm(s_k) * s_k
            else:
                s_k = s_k / M_k

            S[k, :] = s_k

        F_all = unsup_obj(X, S, A, Offsets, lambda_)
        F_obj.append(F_all)
        if len(F_obj) > 1 and abs(F_obj[-1] - F_obj[-2]) / F_obj[-2] < epsilon:
            # print('Updating S: Converged!\n\n')
            return S

    # print('Updating S: Reached max iter.\n\n')
    return S



def unsup_obj(X, S, A, Offsets, lambda_):
    """
    X: n x p
    S: K x q
    A: n x K
    Offsets: n x K

    Returns:
        F: objective value
    """
    n, p = X.shape
    K, q = S.shape

    F = 0
    reconstruction = []

    for i in range(n):
        x = X[i, :]
        shifted_S = op_shift(S, Offsets[i, :], p)
        reconstruction.append(np.dot(A[i, :], shifted_S))
        F += 0.5 * np.linalg.norm(x - np.dot(A[i, :], shifted_S)) ** 2 + lambda_ * np.linalg.norm(A[i, :], 1)

    return F

def reconstruction_err(X, S, A, Offsets):
    """
    X: n x p
    S: K x q
    A: n x K
    Offsets: n x K

    Returns:
        mse : mean squared error
    """
    n, p = X.shape
    K, q = S.shape

    mse = 0
    reconstruction = []

    for i in range(n):
        x = X[i, :]
        shifted_S = op_shift(S, Offsets[i, :], p)
        reconstruction.append(np.dot(A[i, :], shifted_S))
        mse += np.linalg.norm(x - np.dot(A[i, :], shifted_S)) ** 2

    mse /= n
    return mse, reconstruction

def shift_atom(atom, shift, q):
    """Shift atom by padding with zeros."""
    shifted = np.zeros(q)
    if shift + len(atom) <= q:
        shifted[shift:shift + len(atom)] = atom
    else:
        # Truncate if shift would overflow
        end = q - shift
        if end > 0:
            shifted[shift:] = atom[:end]
    return shifted

def compute_mse(X_test, S, A, Offsets):
    n_test, q = X_test.shape
    K = S.shape[0]
    mse = 0.0

    for i in range(n_test):
        reconstruction = np.zeros(q)
        for k in range(K):
            alpha = A[i, k]
            t = Offsets[i, k]
            shifted_atom = shift_atom(S[k], t, q)
            reconstruction += alpha * shifted_atom
        error = X_test[i] - reconstruction
        mse += np.sum(error ** 2)

    mse /= n_test
    return mse

def visualize_results(X, S, A, Offsets, sample_idx=12, plot_dir = "plots"):
    """
    Visualize the reconstruction of a single time series.
    """
    import matplotlib.pyplot as plt
    import os

    # Get the reconstruction
    os.makedirs(plot_dir, exist_ok=True)

    p = X.shape[1]
    shifted_S = op_shift(S, Offsets[sample_idx], p)
    reconstruction = A[sample_idx] @ shifted_S

    # Plot original and reconstruction
    plt.figure(figsize=(12, 6))
    plt.plot(X[sample_idx], label='Original', alpha=0.7)
    plt.plot(reconstruction, label='Reconstruction', alpha=0.7)
    plt.legend()
    plt.title(f'Time Series {sample_idx} Reconstruction')
    plt.savefig(os.path.join(plot_dir, f'reconstruction_{sample_idx}.png'))
    plt.close()

    for k in range(S.shape[0]):
        file_path = os.path.join(plot_dir, f"shapelet_{k}.png")
        plt.figure(figsize=(8, 4))
        plt.plot(S[k], linewidth=2)
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Shapelet {k}')
        plt.savefig(file_path)
        plt.close()

def rank_shapelets(A, top_k=10):
    """
    Ranks shapelets based on their absolute coefficient values across all samples.

    Parameters:
        A (numpy.ndarray): Matrix of shape (n_samples, n_shapelets) containing coefficients.
        top_k (int): Number of top shapelets to consider for frequency ranking.

    Returns:
        sorted_A (numpy.ndarray): Matrix with sorted coefficients per sample.
        sorted_indices (numpy.ndarray): Indices of shapelets in sorted order per sample.
        overall_rank (numpy.ndarray): Shapelet indices ranked by mean importance.
        shapelet_top_k_rank (numpy.ndarray): Shapelet indices ranked by top-k frequency.
        mean_ranks (numpy.ndarray): Mean rank of each shapelet.
        top_k_counts (numpy.ndarray): Count of how often each shapelet appears in top-k.
    """
    n_samples, n_shapelets = A.shape  # Get matrix dimensions

    # Sort each row by absolute value, preserving shapelet indices
    sorted_indices = np.argsort(-np.abs(A), axis=1)  # Sort by descending |coeff|
    sorted_A = np.take_along_axis(A, sorted_indices, axis=1)  # Apply sorting to A

    # Compute mean rank per shapelet
    shapelet_ranks = np.argsort(sorted_indices, axis=1)
    column_names = [f"shapelet_{i}" for i in range(1, 21)]
    df = pd.DataFrame(shapelet_ranks, columns=column_names)
    df.to_csv("shapelet_ranks.csv", index=False)
    # Get ranks for each sample
    mean_ranks = np.mean(shapelet_ranks, axis=0)  # Mean rank across all samples
    overall_rank = np.argsort(mean_ranks)  # Lower rank means more important

    # Compute top-k frequency
    top_k_counts = np.zeros(n_shapelets)
    for i in range(n_samples):
        top_k_indices = sorted_indices[i, :top_k]  # Get top-k shapelets for this sample
        for shapelet in top_k_indices:
            top_k_counts[shapelet] += 1  # Count occurrences

    shapelet_top_k_rank = np.argsort(-top_k_counts)  # Rank shapelets by frequency in top-k

    return sorted_A, sorted_indices, overall_rank, shapelet_top_k_rank, mean_ranks, top_k_counts

def euclidean_distance(sequence_s, sequence_r):
    """
    Calculate the Euclidean distance between two subsequences of equal length.

    Parameters:
    sequence_s (array-like): First subsequence
    sequence_r (array-like): Second subsequence

    Returns:
    float: The Euclidean distance between the subsequences
    """
    # Convert inputs to numpy arrays for efficient calculation
    s = np.array(sequence_s)
    r = np.array(sequence_r)

    # Check if sequences have the same length
    if len(s) != len(r):
        raise ValueError("Sequences must have the same length")

    # Calculate Euclidean distance
    distance = np.sqrt(np.sum((s - r) ** 2))

    return distance

def find_closest_shapelets(shapelets_a, shapelets_b, top_k=1):
    """
    For each shapelet in set A, find the top_k closest shapelets from set B.

    Parameters:
    shapelets_a (list): List of reference shapelets
    shapelets_b (list): List of candidate shapelets
    top_k (int): Number of closest shapelets to find for each shapelet in A

    Returns:
    dict: Dictionary mapping each shapelet index in A to a list of (distance, shapelet_index) tuples
    """
    closest_matches = {}

    for i, shapelet_a in enumerate(shapelets_a):
        # Calculate distances to all shapelets in B
        distances = []
        for j, shapelet_b in enumerate(shapelets_b):
            try:
                dist = euclidean_distance(shapelet_a, shapelet_b)
                distances.append((dist, j))
            except ValueError as e:
                print(f"Error comparing shapelet A[{i}] with B[{j}]: {e}")
                continue

        # Get the top_k closest shapelets
        top_matches = heapq.nsmallest(top_k, distances, key=lambda x: x[0])
        closest_matches[i] = top_matches

    return closest_matches

def combine_shapelet_sets(shapelets_a, shapelets_b, closest_matches, top_per_shapelet=1):
    """
    Combine shapelet set A with the closest matches from set B.

    Parameters:
    shapelets_a (list): List of reference shapelets
    shapelets_b (list): List of candidate shapelets
    closest_matches (dict): Dictionary mapping each shapelet index in A to a list of (distance, shapelet_index) tuples
    top_per_shapelet (int): Number of closest matches to include for each shapelet in A

    Returns:
    list: Combined list of shapelets
    """
    # Start with all shapelets from set A
    combined_shapelets = shapelets_a.copy()

    # Track indices of shapelets from B that we've already added
    added_indices = set()

    # For each shapelet in A, add its closest matches from B
    for i in range(len(shapelets_a)):
        if i in closest_matches:
            for j in range(min(top_per_shapelet, len(closest_matches[i]))):
                _, b_idx = closest_matches[i][j]
                if b_idx not in added_indices:
                    combined_shapelets.append(shapelets_b[b_idx])
                    added_indices.add(b_idx)

    return combined_shapelets

def combine_top_10_shapelets(shapelets_a, shapelets_b):
    """
    Find the 10 closest shapelets from B for set A and combine them.

    Parameters:
    shapelets_a (numpy.ndarray or list): Array of reference shapelets (size 10)
    shapelets_b (numpy.ndarray or list): Array of candidate shapelets (size 20)

    Returns:
    numpy.ndarray: Combined array of shapelets (size 20)
    """
    # Convert to numpy arrays if they aren't already
    shapelets_a = np.array(shapelets_a)
    shapelets_b = np.array(shapelets_b)

    # Calculate all pairwise distances
    all_distances = []

    for i, shapelet_a in enumerate(shapelets_a):
        for j, shapelet_b in enumerate(shapelets_b):
            try:
                dist = euclidean_distance(shapelet_a, shapelet_b)
                all_distances.append((dist, i, j))
            except ValueError as e:
                print(f"Error comparing shapelet A[{i}] with B[{j}]: {e}")
                continue

    # Sort by distance
    all_distances.sort(key=lambda x: x[0])

    # Get indices of top 10 closest shapelets from B
    selected_indices = set()
    top_matches = []

    for dist, i, j in all_distances:
        if j not in selected_indices:
            selected_indices.add(j)
            top_matches.append((dist, j))
            if len(selected_indices) >= 10:
                break

    # Extract the indices of the selected shapelets from B
    selected_b_indices = [j for _, j in top_matches]

    # Combine shapelets using concatenate function
    combined_shapelets = np.concatenate([shapelets_a, shapelets_b[selected_b_indices]], axis=0)

    return combined_shapelets

def combine_top_10_dissimilar_shapelets(shapelets_a, shapelets_b):
    """
    Find the 10 most dissimilar shapelets from B compared to set A and combine them.

    Parameters:
    shapelets_a (numpy.ndarray or list): Array of reference shapelets (size 10)
    shapelets_b (numpy.ndarray or list): Array of candidate shapelets (size 20)

    Returns:
    numpy.ndarray: Combined array of shapelets (size 20)
    """
    # Convert to numpy arrays if they aren't already
    shapelets_a = np.array(shapelets_a)
    shapelets_b = np.array(shapelets_b)

    # For each shapelet in B, find its minimum distance to any shapelet in A
    min_distances = []

    for j, shapelet_b in enumerate(shapelets_b):
        # Calculate distances to all shapelets in A
        distances = []
        for i, shapelet_a in enumerate(shapelets_a):
            try:
                dist = euclidean_distance(shapelet_a, shapelet_b)
                distances.append(dist)
            except ValueError as e:
                print(f"Error comparing shapelet A[{i}] with B[{j}]: {e}")
                continue

        # Find the minimum distance to any shapelet in A
        if distances:
            min_dist = min(distances)
            min_distances.append((min_dist, j))
        else:
            print(f"Warning: No valid distances for shapelet B[{j}]")

    # Sort by distance in descending order to get the most dissimilar first
    min_distances.sort(key=lambda x: x[0], reverse=True)

    # Get indices of the 10 most dissimilar shapelets from B
    selected_b_indices = [j for _, j in min_distances[:10]]

    # Combine shapelets using numpy's concatenate function
    combined_shapelets = np.concatenate([shapelets_a, shapelets_b[selected_b_indices]], axis=0)

    return combined_shapelets

def shapelet_distance(time_series, shapelet):
    """
    Compute the minimum Euclidean distance between a shapelet and all subsequences of the time series.

    Parameters:
    - time_series (numpy.ndarray): The full time series of length N.
    - shapelet (numpy.ndarray): The shapelet of length L.

    Returns:
    - float: Minimum Euclidean distance between shapelet and time series.
    """
    N = len(time_series)
    L = len(shapelet)
    min_dist = float('inf')  # Initialize with a large value

    # Slide the shapelet across all possible positions in the time series
    for i in range(N - L + 1):
        subsequence = time_series[i:i + L]  # Extract subsequence of same length as shapelet
        dist = euclidean(subsequence, shapelet)  # Compute Euclidean distance
        min_dist = min(min_dist, dist)  # Keep the minimum distance

    return min_dist

def shapelet_transform(dataset, shapelets):
    """
    Transforms a dataset of time series using shapelets.

    Parameters:
    - dataset (numpy.ndarray): Time series dataset of shape (num_samples, time_series_length).
    - shapelets (numpy.ndarray): Shapelet set of shape (num_shapelets, shapelet_length).

    Returns:
    - numpy.ndarray: Transformed dataset of shape (num_samples, num_shapelets).
    """
    num_samples = dataset.shape[0]
    num_shapelets = shapelets.shape[0]
    transformed_features = np.zeros((num_samples, num_shapelets))

    for i, time_series in enumerate(dataset):
        for j, shapelet in enumerate(shapelets):
            transformed_features[i, j] = shapelet_distance(time_series, shapelet)

    return transformed_features

def learn_shapelets(train_x, train_y, K=20, lambdas =0.1, r=0.25):
    """
        Extract shapelets using shift invariant dictionary learning method.

        Parameters:
        - train x (numpy.ndarray): Time series dataset of shape (num_samples, time_series_length).
        - train y (numpy.ndarray): Associated label
        - K: number of shapelets
        - lambdas: regularisation factor
        - r: q/p ratio of length of shapelet to length of time series


        Returns:
        - numpy.ndarray: Set of learnt shapelets as basis of dictionary.
        """

    c = 100
    epsilon = 1e-5
    maxIter = 1e3
    maxInnerIter = 5

    n_train, p = train_x.shape
    q = int(np.ceil(p * r))
    runid = f'l_{lambdas}_K_{K}_q_{q}'

    # Train SIDL on train dataset
    start_time =  time.time()
    S, A, Offsets, F_obj = USIDL(train_x, train_y, lambdas, K, q, c, epsilon, maxIter, maxInnerIter, runid)
    print("Time taken to learn shapelets in seconds - ", time.time() - start_time)


    return S


def learn_shapelets_alpha_const(train_x, train_y, K=20, lambdas =0.1, r=0.25):
    """
        Extract shapelets using shift invariant dictionary learning method.

        Parameters:
        - train x (numpy.ndarray): Time series dataset of shape (num_samples, time_series_length).
        - train y (numpy.ndarray): Associated label
        - K: number of shapelets
        - lambdas: regularisation factor
        - r: q/p ratio of length of shapelet to length of time series


        Returns:
        - numpy.ndarray: Set of learnt shapelets as basis of dictionary.
        """

    c = 100
    epsilon = 1e-5
    maxIter = 1e3
    maxInnerIter = 5

    n_train, p = train_x.shape
    q = int(np.ceil(p * r))
    runid = f'l_{lambdas}_K_{K}_q_{q}'

    # Train SIDL on train dataset
    start_time =  time.time()
    S, A, Offsets, F_obj = USIDL_with_alpha_const(train_x, train_y, lambdas, K, q, c, epsilon, maxIter, maxInnerIter, runid)
    print("Time taken to learn shapelets in seconds - ", time.time() - start_time)


    return S, A, Offsets, F_obj


def init_dictionary_with_svd(X, K, q):
    """
    X: array of shape (n_samples, ts_length)
    K: number of shapelets (K)
    shapelet_length: length of each shapelet (q)
    """
    S = []
    for _ in range(K):
        # Randomly select a signal and a start index
        idx = np.random.randint(0, X.shape[0])
        start = np.random.randint(0, X.shape[1] - q)
        chunk = X[idx, start:start + q].reshape(1, -1)

        # SVD for rank-1 approximation
        u, s, vh = np.linalg.svd(chunk, full_matrices=False)
        rank1_atom = s[0] * vh[0]  # (1,) * (q,) -> (q,)
        S.append(rank1_atom)

    S = np.array(S)
    return S

def init_dictionary_with_random_chunks(X, K, q):
    """
        X: array of shape (n_samples, ts_length)
        K: number of shapelets (K)
        shapelet_length: length of each shapelet (q)

        Returns: random chunks from train data as initial shapelet dictionary
        """

    S = []
    for _ in range(K):
        sample_idx = np.random.randint(0, X.shape[0])
        max_start = X.shape[1] - q
        start_idx = np.random.randint(0, max_start + 1)
        chunk = X[sample_idx, start_idx:start_idx + q]
        S.append(chunk)

    return np.array(S)

def learn_shapelet_dict_via_sidl(X_batch):

    "Batch data with labels to learn labelwise subdictionaries"

    train_batch = np.asarray(X_batch)
    # train_batch = X_batch.to_numpy()
    train_labels = train_batch[:, -1]
    shapelets = {}
    # re = {}

    for label in np.unique(train_labels):
        class_data = train_batch[np.where(train_labels == label)[0]]
        class_labels = train_labels[np.where(train_labels == label)[0]]
        # np.savetxt(f'{label}_train_data_for_shapelets.csv', class_data)
        print(f"Learning shapelets for label - {label}")
        train_data = class_data[:, :-1].astype(np.float64)
        S, A, offsets, F_obj = learn_shapelets_alpha_const(train_data, class_labels, K=10, lambdas=1, r=0.25)
        # recon_err, _ = reconstruction_err(train_data, S, A, offsets)
        # re[label] = recon_err
        shapelets[label] = S

    return shapelets

def re_for_threshold(X, S):

    n = X.shape[0]
    p = X.shape[1]
    S = np.array(S)
    K = S.shape[0]
    q = S.shape[1]
    A = np.random.rand(n, K)  # initializations of alpha change to allow only positive values
    offsets = np.random.randint(0, p - q + 1, (n, K))
    A_batch, offsets_batch, F = update_A_par_with_alpha_const(X, S, A, offsets, lambda_=1, epsilon=1e-5,
                                                              maxIter=1e3)
    re, reconstructed_signals = reconstruction_err(X, S, A_batch, offsets_batch)
    # plt.plot(X[0])
    # plt.plot(reconstructed_signals[0])

    return re, A_batch

def re_for_threshold_labelwise(X, S):


    train_batch = np.asarray(X)
    # train_batch = X_batch.to_numpy()
    train_labels = train_batch[:, -1]
    re = {}

    for label in np.unique(train_labels):
        class_data = train_batch[np.where(train_labels == label)[0]]
        # class_labels = train_labels[np.where(train_labels == label)[0]]
        train_data = class_data[:, :-1].astype(np.float64)
        shapelets = S[label]
        n = train_data.shape[0]
        p = train_data.shape[1]
        S = np.asarray(shapelets)
        K = S.shape[0]
        q = S.shape[1]

        A = np.random.rand(n, K)  # initializations of alpha change to allow only positive values
        offsets = np.random.randint(0, p - q + 1, (n, K))
        A_batch, offsets_batch, F = update_A_par_with_alpha_const(train_data, S, A, offsets, lambda_=1, epsilon=1e-5,
                                                                  maxIter=1e3)
        recons_err , reconstructed_signals = reconstruction_err(train_data, S, A_batch, offsets_batch)
        # plt.plot(train_data[0])
        # plt.plot(reconstructed_signals[0])
        re[label] = recons_err

        return re

def combine_shapelets_w_sparse_coeff(A_old, A_new, shapelets_old, shapelets_new, train_data_w_label):

    A_old = np.asarray(A_old)
    A_new = np.asarray(A_new)
    A_old_df = pd.DataFrame(A_old)
    A_new_df = pd.DataFrame(A_new)

    n_samples, n_shapelets = A_old.shape

    labels = train_data_w_label[:,-1].reshape(-1, 1)
    k = n_shapelets // len(np.unique(labels))
    A_old_df["label"] = labels
    A_new_df["label"] = labels
    updated_shapelets_dict = {}
    dictionary_with_source = {}

    for label in np.unique(labels):
        label_df_old = A_old_df[A_old_df["label"] == label]
        indices_old = [i for i, (lbl, _) in enumerate(shapelets_old) if lbl == label]
        sum_alpha_old = label_df_old[indices_old].sum(axis=0)
        label_df_new = A_new_df[A_new_df["label"] == label]
        indices_new = [i for i, (lbl, _) in enumerate(shapelets_new) if lbl == label]
        sum_alpha_new = label_df_new[indices_new].sum(axis=0)

    #    combine the series and get top k shapelets from old and new based on the highest coefficient value obtained after sum (across all time samples of that class)
        indexed = [(v, "old_coeff", i) for i, v in enumerate(sum_alpha_old)] + [(v, "new_coeff", i) for i, v in enumerate(sum_alpha_new)]

        top_k = heapq.nlargest(k, indexed)
        updated_shapelets_dict[label] = []
        dictionary_with_source[label] = []
        for shapelet_coeff, shap, idx in top_k:
            if shap == "new_coeff":
                shapelet_new = [shapelet for lbl, shapelet in shapelets_new if lbl == label]
                shap_new = shapelet_new[idx]
                updated_shapelets_dict[label].append(shap_new)
                dictionary_with_source[label].append(("new_coeff", shap_new))
            else:
                shapelet_old = [shapelet for lbl, shapelet in shapelets_old if lbl == label]
                shap_old = shapelet_old[idx]
                updated_shapelets_dict[label].append(shap_old)
                dictionary_with_source[label].append(("old_coeff", shap_old))

        # for val, src, idx in top_k:
        #     print(f"Value: {val:.4f} from {src} at index {idx}")


    return updated_shapelets_dict, dictionary_with_source









