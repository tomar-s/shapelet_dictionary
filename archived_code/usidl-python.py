import numpy as np
from typing import Tuple, List
import time

# def op_shift(S: np.ndarray, offsets: np.ndarray, target_dim: int) -> np.ndarray:
#     """
#     Shift the shapelets according to given offsets.
#
#     Args:
#         S: Shapelet matrix of shape (K, q)
#         offsets: Offset vector of shape (K,)
#         target_dim: Target dimension for the shifted matrix
#
#     Returns:
#         Shifted shapelet matrix of shape (K, target_dim)
#     """
#     K, q = S.shape
#     res = np.zeros((target_dim, K))
#
#     # Equivalent to MATLAB: repmat(offsets+1, q, 1)
#     IDX = np.tile(offsets + 1, (q, 1))
#
#     # Equivalent to MATLAB: bsxfun(@plus, IDX, [0:q-1]')
#     IDX = IDX + np.arange(q)[:, np.newaxis]
#
#     # Equivalent to MATLAB: bsxfun(@plus, IDX, [0:(K-1)] * target_dim)
#     IDX = IDX + (np.arange(K) * target_dim)
#
#     # Convert to linear indices
#     IDX = IDX.ravel(order='F') - 1  # Subtract 1 for 0-based indexing
#
#     # Equivalent to MATLAB: res(IDX) = S'
#     res.ravel(order='F')[IDX] = S.T.ravel(order='F')
#
#     # Return transposed result
#     return res.T

def op_shift(S: np.ndarray, offsets: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Shift the shapelets according to given offsets.

    Args:
        S: Shapelet matrix of shape (K, q)
        offsets: Offset vector of shape (K,)
        target_dim: Target dimension for the shifted matrix

    Returns:
        Shifted shapelet matrix of shape (K, target_dim)
    """
    K, q = S.shape
    res = np.zeros((K, target_dim))

    # MATLAB: res(IDX) = S';
    # In Python, we'll do this directly by correctly placing each shapelet
    for k in range(K):
        if offsets[k] + q <= target_dim:
            res[k, offsets[k]:(offsets[k] + q)] = S[k, :]

    return res


def unsup_obj(X: np.ndarray, S: np.ndarray, A: np.ndarray,
              Offsets: np.ndarray, lambda_: float) -> float:
    """
    Compute the unsupervised objective function value.
    """
    n, p = X.shape
    F = 0.0

    # Calculate reconstruction error for each time series
    reconstruction_errors = []
    for i in range(n):
        x = X[i]
        shifted_S = op_shift(S, Offsets[i], p)
        reconstruction = A[i] @ shifted_S
        error = np.mean((x - reconstruction) ** 2)
        reconstruction_errors.append(error)
        F += 0.5 * error + lambda_ * np.sum(np.abs(A[i]))

    # Print some statistics about the reconstruction
    mean_error = np.mean(reconstruction_errors)
    max_error = np.max(reconstruction_errors)
    min_error = np.min(reconstruction_errors)

    if np.random.random() < 0.01:  # Print occasionally to avoid too much output
        print(f"\nReconstruction error statistics:")
        print(f"Mean error: {mean_error:.6f}")
        print(f"Max error: {max_error:.6f}")
        print(f"Min error: {min_error:.6f}")
        print(f"L1 penalty term: {lambda_ * np.sum(np.abs(A)):.6f}")

    return F

# def unsup_obj(X: np.ndarray, S: np.ndarray, A: np.ndarray,
#               Offsets: np.ndarray, lambda_: float) -> float:
#     """
#     Compute the unsupervised objective function value.
#
#     Args:
#         X: Time series matrix of shape (n, p)
#         S: Shapelet matrix of shape (K, q)
#         A: Coefficient matrix of shape (n, K)
#         Offsets: Offset matrix of shape (n, K)
#         lambda_: L1 regularization parameter
#
#     Returns:
#         Objective function value
#     """
#     n, p = X.shape
#     F = 0.0
#
#     for i in range(n):
#         x = X[i]
#         shifted_S = op_shift(S, Offsets[i], p)
#         reconstruction = A[i] @ shifted_S
#         F += 0.5 * np.sum((x - reconstruction) ** 2) + lambda_ * np.sum(np.abs(A[i]))
#
#     return F


def update_A_par(X: np.ndarray, S: np.ndarray, A: np.ndarray,
                 Offsets: np.ndarray, lambda_: float,
                 maxIter: int, epsilon: float) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Update the coefficient matrix A and corresponding offsets with numerical stability improvements.
    """
    n, p = X.shape
    K, q = S.shape
    seg_idx = np.array([list(range(j, j + q)) for j in range(p - q + 1)])

    F_obj = []
    initial_error = unsup_obj(X, S, A, Offsets, lambda_)
    print(f"Initial reconstruction error: {initial_error:.6f}")

    # Normalize S to prevent numerical issues
    S_norms = np.linalg.norm(S, axis=1)
    S_normalized = S / (S_norms[:, np.newaxis] + 1e-10)

    min_improvement = 1e-6  # Minimum relative improvement required
    prev_error = float('inf')

    for iter in range(maxIter):
        old_A = A.copy()
        old_Offsets = Offsets.copy()

        for i in range(n):
            x = X[i]
            # Normalize input to prevent numerical issues
            x_norm = np.linalg.norm(x)
            if x_norm > 0:
                x_normalized = x / x_norm
            else:
                continue

            offs = Offsets[i]
            shifted_S = op_shift(S_normalized, offs, p)

            for k in np.random.permutation(K):
                base = S_normalized[k]
                temp_a = A[i].copy()
                temp_a[k] = 0

                x_residue = x_normalized - temp_a @ shifted_S
                base_norm2 = np.sum(base ** 2)

                if base_norm2 < 1e-10:  # Skip if base is essentially zero
                    continue

                segs = np.array([x_residue[idx] for idx in seg_idx])
                dot_prods = segs @ base

                M_dp = np.max(np.abs(dot_prods))
                M_idx = np.argmax(np.abs(dot_prods))

                if M_dp <= lambda_:
                    a_k_star = 0
                else:
                    # Denormalize the coefficient
                    a_k_star = np.sign(dot_prods[M_idx]) * (M_dp - lambda_) / base_norm2
                    a_k_star = a_k_star * x_norm / (S_norms[k] + 1e-10)
                    t_k_star = M_idx
                    Offsets[i, k] = t_k_star

                # Add stability check
                if not np.isfinite(a_k_star):
                    a_k_star = 0

                A[i, k] = a_k_star

        # Calculate changes
        A_change = np.max(np.abs(A - old_A))
        Offsets_change = np.max(np.abs(Offsets - old_Offsets))

        F_all = unsup_obj(X, S, A, Offsets, lambda_)
        F_obj.append(F_all)

        if iter % 10 == 0:
            print(f"Iteration {iter}")
            print(f"Max A change: {A_change:.6f}")
            print(f"Max Offsets change: {Offsets_change}")
            print(f"Current error: {F_all:.6f}")

        # More stringent convergence criterion
        rel_improvement = abs(prev_error - F_all) / (abs(prev_error) + 1e-10)
        if rel_improvement < min_improvement and iter > 10:
            print(f"Converged at iteration {iter}")
            break

        prev_error = F_all

    print(f"Final reconstruction error: {F_all:.6f}")
    print(f"Number of non-zero coefficients: {np.sum(np.abs(A) > 1e-10)}")
    return A, Offsets, F_obj

# def update_A_par(X: np.ndarray, S: np.ndarray, A: np.ndarray,
#                  Offsets: np.ndarray, lambda_: float,
#                  maxIter: int, epsilon: float) -> Tuple[np.ndarray, np.ndarray, List[float]]:
#     """
#     Update the coefficient matrix A and corresponding offsets.
#     """
#     n, p = X.shape
#     K, q = S.shape
#     seg_idx = np.array([list(range(j, j + q)) for j in range(p - q + 1)])
#
#     F_obj = []
#     initial_error = unsup_obj(X, S, A, Offsets, lambda_)
#     print(f"Initial reconstruction error: {initial_error}")
#
#     for iter in range(maxIter):
#         old_A = A.copy()
#         old_Offsets = Offsets.copy()
#
#         for i in range(n):
#             x = X[i]
#             offs = Offsets[i]
#             shifted_S = op_shift(S, offs, p)
#
#             for k in np.random.permutation(K):
#                 base = S[k]
#                 temp_a = A[i].copy()
#                 temp_a[k] = 0
#
#                 x_residue = x - temp_a @ shifted_S
#                 base_norm2 = np.sum(base ** 2)
#
#                 # Calculate dot products for all possible alignments
#                 segs = np.array([x_residue[idx] for idx in seg_idx])
#                 dot_prods = segs @ base
#
#                 M_dp = np.max(np.abs(dot_prods))
#                 M_idx = np.argmax(np.abs(dot_prods))
#
#                 if M_dp <= lambda_:
#                     a_k_star = 0
#                 else:
#                     a_k_star = np.sign(dot_prods[M_idx]) * (M_dp - lambda_) / base_norm2
#                     t_k_star = M_idx
#                     Offsets[i, k] = t_k_star
#
#                 A[i, k] = a_k_star
#
#         # Calculate changes in A and Offsets
#         A_change = np.max(np.abs(A - old_A))
#         Offsets_change = np.max(np.abs(Offsets - old_Offsets))
#
#         F_all = unsup_obj(X, S, A, Offsets, lambda_)
#         F_obj.append(F_all)
#
#         if iter % 10 == 0:  # Print every 10 iterations
#             print(f"Iteration {iter}")
#             print(f"Max A change: {A_change}")
#             print(f"Max Offsets change: {Offsets_change}")
#             print(f"Current error: {F_all}")
#
#         if len(F_obj) > 1 and abs(F_obj[-1] - F_obj[-2]) / abs(F_obj[-2]) < epsilon:
#             print(f"Converged at iteration {iter}")
#             break
#
#     print(f"Final reconstruction error: {F_obj[-1]}")
#     print(f"Number of non-zero coefficients: {np.sum(np.abs(A) > 1e-10)}")
#     return A, Offsets, F_obj


# def update_A_par(X: np.ndarray, S: np.ndarray, A: np.ndarray,
#                  Offsets: np.ndarray, lambda_: float,
#                  maxIter: int, epsilon: float) -> Tuple[np.ndarray, np.ndarray, List[float]]:
#     """
#     Update the coefficient matrix A and corresponding offsets.
#     """
#     n, p = X.shape
#     K, q = S.shape
#     seg_idx = np.array([list(range(j, j + q)) for j in range(p - q + 1)])
#
#     F_obj = []
#
#     for iter in range(maxIter):
#         for i in range(n):
#             x = X[i]
#             offs = Offsets[i]
#             shifted_S = op_shift(S, offs, p)
#
#             for k in np.random.permutation(K):
#                 base = S[k]
#                 temp_a = A[i].copy()
#                 temp_a[k] = 0
#
#                 x_residue = x - temp_a @ shifted_S
#                 residue_norm2 = np.sum(x_residue ** 2)
#                 base_norm2 = np.sum(base ** 2)
#
#                 segs = np.array([x_residue[idx] for idx in seg_idx])
#                 dot_prods = segs @ base
#
#                 M_dp = np.max(np.abs(dot_prods))
#                 M_idx = np.argmax(np.abs(dot_prods))
#
#                 if M_dp <= lambda_:
#                     a_k_star = 0
#                     Offsets[i, k] = 0
#                 else:
#                     a_k_star = np.sign(dot_prods[M_idx]) * (M_dp - lambda_) / base_norm2
#                     t_k_star = M_idx
#                     Offsets[i, k] = t_k_star
#
#                 A[i, k] = a_k_star
#
#         F_all = unsup_obj(X, S, A, Offsets, lambda_)
#         F_obj.append(F_all)
#
#         if len(F_obj) > 1 and abs(F_obj[-1] - F_obj[-2]) / F_obj[-2] < epsilon:
#             break
#
#     return A, Offsets, F_obj

def update_S(X: np.ndarray, S: np.ndarray, A: np.ndarray, 
            Offsets: np.ndarray, lambda_: float, c: float, 
            maxIter: int, epsilon: float) -> np.ndarray:
    """
    Update the shapelet dictionary S.
    """
    n, p = X.shape
    K, q = S.shape
    F_obj = []
    
    for iter in range(maxIter):
        for k in range(K):
            M_k = np.sum(A[:, k] ** 2)
            if M_k == 0:
                continue
                
            s_k = np.zeros(q)
            
            for i in range(n):
                temp_a = A[i].copy()
                temp_a[k] = 0
                shifted_S = op_shift(S, Offsets[i], p)
                xi_residue = X[i] - temp_a @ shifted_S
                
                t_ik = Offsets[i, k]
                if t_ik + q <= p:
                    s_k += A[i, k] * xi_residue[t_ik:t_ik + q]
            
            s_k_norm = np.linalg.norm(s_k)
            if M_k <= s_k_norm / np.sqrt(c):
                s_k = np.sqrt(c) / s_k_norm * s_k
            else:
                s_k = s_k / M_k
            
            S[k] = s_k
        
        F_all = unsup_obj(X, S, A, Offsets, lambda_)
        F_obj.append(F_all)
        
        if len(F_obj) > 1 and abs(F_obj[-1] - F_obj[-2]) / F_obj[-2] < epsilon:
            break
            
    return S

def visualize_results(X, S, A, Offsets, sample_idx=0):
    """
    Visualize the reconstruction of a single time series.
    """
    import matplotlib.pyplot as plt

    # Get the reconstruction
    p = X.shape[1]
    shifted_S = op_shift(S, Offsets[sample_idx], p)
    reconstruction = A[sample_idx] @ shifted_S

    # Plot original and reconstruction
    plt.figure(figsize=(12, 6))
    plt.plot(X[sample_idx], label='Original', alpha=0.7)
    plt.plot(reconstruction, label='Reconstruction', alpha=0.7)
    plt.legend()
    plt.title(f'Time Series {sample_idx} Reconstruction')
    plt.savefig(f'reconstruction_{sample_idx}.png')
    plt.close()

    # Plot learned shapelets
    plt.figure(figsize=(12, 6))
    for k in range(min(5, S.shape[0])):  # Plot first 5 shapelets
        plt.subplot(5, 1, k + 1)
        plt.plot(S[k])
        plt.title(f'Shapelet {k}')
    plt.tight_layout()
    plt.savefig('shapelets.png')
    plt.close()


def USIDL(X: np.ndarray, y: np.ndarray, lambda_: float, K: int, q: int,
          c: float, epsilon: float, maxIter: int, maxInnerIter: int,
          runid: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    Improved USIDL algorithm implementation with better initialization and stability.
    """
    n, p = X.shape

    # Better initialization
    np.random.seed(1)

    # Initialize S using random segments from the data
    S = np.zeros((K, q))
    for k in range(K):
        # Randomly select a time series and starting position
        i = np.random.randint(0, n)
        start = np.random.randint(0, p - q + 1)
        S[k] = X[i, start:start + q]

    # Normalize S
    S_norms = np.linalg.norm(S, axis=1)
    S = S / (S_norms[:, np.newaxis] + 1e-10)

    # Initialize A with small random values
    A = np.random.randn(n, K) * 0.01

    # Initialize Offsets more sensibly
    Offsets = np.random.randint(0, p - q + 1, size=(n, K))

    F_obj = []
    prev_error = float('inf')
    min_improvement = 1e-6

    for iter in range(maxIter):
        # Update coefficients and matching offsets
        A, Offsets, _ = update_A_par(X, S, A, Offsets, lambda_, maxInnerIter, epsilon)

        # Update bases
        S = update_S(X, S, A, Offsets, lambda_, c, maxInnerIter, epsilon)

        # Check convergence
        F_all = unsup_obj(X, S, A, Offsets, lambda_)
        F_obj.append(F_all)

        # Print progress
        if iter % 5 == 0:
            print(f"Outer iteration {iter}, objective: {F_all:.6f}")

        # More stringent convergence check
        rel_improvement = abs(prev_error - F_all) / (abs(prev_error) + 1e-10)
        if rel_improvement < min_improvement and iter > 10:
            print('Converged!')
            break

        prev_error = F_all

    return S, A, Offsets, F_obj

# def USIDL(X: np.ndarray, y: np.ndarray, lambda_: float, K: int, q: int,
#           c: float, epsilon: float, maxIter: int, maxInnerIter: int,
#           runid: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
#     """
#     Main USIDL algorithm implementation.
#
#     Args:
#         X: Time series matrix of shape (n, p)
#         y: Labels (not used in unsupervised version)
#         lambda_: L1 regularization parameter
#         K: Number of shapelets
#         q: Length of shapelets
#         c: Squared L2-norm constraint for shapelets
#         epsilon: Convergence threshold
#         maxIter: Maximum number of outer iterations
#         maxInnerIter: Maximum number of inner iterations
#         runid: Run identifier
#
#     Returns:
#         S: Learned shapelet dictionary
#         A: Learned coefficients
#         Offsets: Learned offsets
#         F_obj: Objective function values
#     """
#     n, p = X.shape
#
#     # Initialize parameters
#     np.random.seed(1)  # For reproducibility
#     S = np.random.randn(K, q)
#     A = np.random.randn(n, K)
#     Offsets = np.random.randint(0, p - q + 1, size=(n, K))
#
#     F_obj = []
#
#     for iter in range(maxIter):
#         # Update coefficients and matching offsets
#         A, Offsets, _ = update_A_par(X, S, A, Offsets, lambda_, maxInnerIter, epsilon)
#
#         # Update bases
#         S = update_S(X, S, A, Offsets, lambda_, c, maxInnerIter, epsilon)
#
#         # Check convergence
#         F_all = unsup_obj(X, S, A, Offsets, lambda_)
#         F_obj.append(F_all)
#
#         if len(F_obj) > 1 and abs(F_obj[-1] - F_obj[-2]) / F_obj[-2] < epsilon:
#             print('Converged!')
#             break
#
#     if iter == maxIter - 1:
#         print('Maximum Iteration Reached!')
#
#     return S, A, Offsets, F_obj

# Example usage
def main():
    """
    Main function implementing the complete training and testing pipeline with grid search.
    """
    np.random.seed(1)

    # Dataset loading
    dataset_name = 'bike_data'

    # train_data = np.loadtxt("/Users/shivanitomar/Documents/Implementations/TTM/shapelet_matlab_data_files/orig_train_256_bike_data.csv", delimiter=",")
    # test_data = np.loadtxt("/Users/shivanitomar/Documents/Implementations/TTM/shapelet_matlab_data_files/orig_test_256_bike_data.csv", delimiter=",")
    train_data = np.loadtxt("/Users/shivanitomar/Downloads/GunPoint_TRAIN (1).tsv", delimiter="\t")
    test_data = np.loadtxt("/Users/shivanitomar/Downloads/GunPoint_TEST (1).tsv", delimiter="\t")
    # train_data = np.loadtxt(f'{dataset_name}_TRAIN')
    # test_data = np.loadtxt(f'{dataset_name}_TEST')

    # Split into features and labels
    train_X = train_data[:, 1:]
    test_X = test_data[:, 1:]
    train_y = train_data[:, 0]
    test_y = test_data[:, 0]

    n_train, p = train_X.shape
    n_test, _ = test_X.shape

    # Algorithm parameters
    c = 100
    epsilon = 1e-5
    maxIter = int(1e3)
    maxInnerIter = 5

    # Grid search parameters
    Ks = [20]  # [20, 50, 100]
    lambdas = [0.01]  # [0.1, 1, 10, 100]
    rs = [0.1]  # [0.25, 0.5]

    # Grid search
    for K in Ks:
        for lambda_ in lambdas:
            # Initialize test set coefficients
            A_rand_init = np.random.randn(n_test, K)

            for r in rs:
                q = int(np.ceil(p * r))
                # Create run identifier
                runid = f"{dataset_name}_l_{lambda_}_K_{K}_q_{q}"

                # Train SIDL on training set
                start_time = time.time()
                S, A, Offsets, F_obj = USIDL(train_X, train_y, lambda_, K, q, c,
                                             epsilon, maxIter, maxInnerIter, runid)
                learn_time = time.time() - start_time
                # print(S)

                print(f"\n##### TRAINING TIME on TRAIN SET (K={K}, lambda={lambda_}, r={r}): {learn_time:.2f} secs.\n")

                # Learn sparse coding on test set with dictionary learned from training set
                A_test = A_rand_init.copy()
                Offsets_test = np.random.randint(0, p - q + 1, size=(n_test, K))

                start_time = time.time()
                A_test, Offsets_test, F_all = update_A_par(test_X, S, A_test,
                                                           Offsets_test, lambda_,
                                                           maxIter, epsilon)
                fit_time = time.time() - start_time

                visualize_results(test_X, S, A_test, Offsets_test)

                # Get reconstruction error for SIDL
                test_recons_error_sidl = unsup_obj(test_X, S, A_test,
                                                   Offsets_test, 0) / n_test

                print(f"\n##### RECONS ERROR on TEST SET (K={K}, lambda={lambda_}) "
                      f"SIDL (r={r}): {test_recons_error_sidl:.6f}\n")

                # Save results
                results = {
                    'S': S,
                    'A': A,
                    'Offsets': Offsets,
                    'A_test': A_test,
                    'Offsets_test': Offsets_test,
                    'parameters': {
                        'K': K,
                        'lambda': lambda_,
                        'r': r,
                        'q': q,
                        'c': c,
                        'epsilon': epsilon,
                        'maxIter': maxIter,
                        'maxInnerIter': maxInnerIter
                    },
                    'metrics': {
                        'learn_time': learn_time,
                        'fit_time': fit_time,
                        'test_recons_error': test_recons_error_sidl
                    }
                }
                np.save(f"{runid}.npy", results)




if __name__ == "__main__":
    main()
