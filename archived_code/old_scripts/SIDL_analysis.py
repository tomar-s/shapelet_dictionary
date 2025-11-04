import os
import numpy as np
import pandas as pd
import time
import heapq
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
        F_all, _ = unsup_obj(X, S, A, Offsets, lambda_)

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

    S = np.random.randn(K, q)  # initialize bases
    A = np.random.rand(n, K)  # initializations of alpha change to allow only positive values
    Offsets = np.random.randint(0, p - q + 1, (n, K))  # initialize offsets

    F_obj = []

    for iter in range(int(maxIter)):
        # update coefficients and matching offsets
        A, Offsets, F_all = update_A_par_with_alpha_const(X, S, A, Offsets, lambda_, maxInnerIter, epsilon)

        # update bases
        S = update_S(X, S, A, Offsets, lambda_, c, maxInnerIter, epsilon)

        # check convergence
        F_all, _ = unsup_obj(X, S, A, Offsets, lambda_)

        F_obj.append(F_all)
        if len(F_obj) > 1 and abs(F_obj[-1] - F_obj[-2]) / F_obj[-2] < epsilon:
            print('Converged!')
            return S, A, Offsets, F_obj

    print('Maximum Iteration Reached!')
    return S, A, Offsets, F_obj


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

# Example usage:
# S = np.random.randn(5, 10)  # Example basis functions (5 x 10)
# offsets = np.random.randint(0, 90, 5)  # Example offsets (1 x 5)
# target_dim = 100
# shifted_S = op_shift(S, offsets, target_dim)
# print(shifted_S)

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

        F_all, _ = unsup_obj(X, S, A, Offsets, lambda_)
        F_obj.append(F_all)
        if len(F_obj) > 1 and abs(F_obj[-1] - F_obj[-2]) / F_obj[-2] < epsilon:
            # print('Updating A: Converged!\n\n')
            return A, Offsets, F_all

    # print('Updating A: Reached max iter.\n\n')
    return A, Offsets, F_all


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

        F_all, _ = unsup_obj(X, S, A, Offsets, lambda_)
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

            s_k = np.zeros(q)

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

        F_all, _ = unsup_obj(X, S, A, Offsets, lambda_)
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

    return F, reconstruction

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

def initialize_results_file_tsg(csv_file_path):

    if os.path.exists(csv_file_path):
        df_results = pd.read_csv(csv_file_path)
    else:
        df_results = pd.DataFrame(columns=[
            'Alpha_constraint',
            'C', 'Lambda', 'MaxInnerIteration', 'RE'
        ])
    return df_results


# Function to save results to CSV
def save_results(df_results, csv_file_path, result_row):
    row_count = len(df_results) + 1
    df_results.loc[row_count] = result_row
    df_results.to_csv(csv_file_path, index=False)


# Set random seed
np.random.seed(10)

# This is for loading GunPoint data for replicating the SIDL paper results
dataset_name = 'GunPoint'
data = loadmat("/datasets/preliminary/GunPoint/GP_data_array.mat")
train_table = data["train_table"]
test_table = data["test_table"]

train_X = train_table[:, 1:]
test_X  = test_table[:, 1:]
np.savetxt("original_xtest.csv", list(test_X))
train_y = train_table[:, 0]   # Select the first column (index 0)
test_y  = test_table[:, 0]
n_test, p = test_X.shape

# #trying a simple example to understand dictionary learning
# dataset_name = 'Random'
# x = np.random.randn(10, 20)
# y = np.random.randn (10)
C = [1]   #[1, 50, 100, 150]
epsilon = 1e-5
maxIter = 1e3
maxInnerIter_list = [5] #[5, 10, 20]
K=10
q=38
lambdas = [0.1] #[0.1, 1]

results = []

# result = {'Alpha_constraint': None,
#             'C': None,
#             'Lambda': None,
#             'MaxInnerIteration': None,
#             'RE': None}
for c in C:
    for lambda_ in lambdas:
        for maxInnerIter in maxInnerIter_list:
            runid = f'{dataset_name}_l_{lambda_}_K_{K}_q_{q}'
            S_with_alpha_const, A, Offsets, F_obj = USIDL_with_alpha_const(train_X, train_y, lambda_, K, q, c, epsilon, maxIter, maxInnerIter, runid)
            # print("Learnt shapelets with alpha positive constraint", S_with_alpha_const)
            # print("this is a trial1")

            # recontruction error on test
            A_rand_init = np.random.rand(n_test, K)
            A_test_w = A_rand_init

            Offsets_test_w = np.random.randint(0, p - q, (n_test, K))
            A_test_w, Offsets_test_w, F_all_1 = update_A_par_with_alpha_const(test_X, S_with_alpha_const, A_test_w, Offsets_test_w, lambda_, maxIter,
                                                               epsilon)

            test_recons_error_sidl_with_alpha_raw, reconstructed_signals = unsup_obj(test_X, S_with_alpha_const, A_test_w, Offsets_test_w, 0)
            test_recons_error_sidl_with_alpha = test_recons_error_sidl_with_alpha_raw / n_test
            np.savetxt('reconstructed_xtest_method1.csv', reconstructed_signals)
            print(f'\n\n RECONS ERROR on test set with alpha constraint (C={c}, lambda={lambda_}) (maxInnerIter={maxInnerIter}): {test_recons_error_sidl_with_alpha}.\n\n')
            re , reconstructed_signals_re = reconstruction_err(test_X, S_with_alpha_const, A_test_w, Offsets_test_w)
            print("Reconstruction error(MSE) is", re)
            np.savetxt('reconstructed_xtest_method2.csv', reconstructed_signals_re)
            df_original = pd.DataFrame(list(test_X))
            df_reconstruction = pd.DataFrame(reconstructed_signals)
            df_reconstruction_re = pd.DataFrame(reconstructed_signals_re)

            for i in range(df_original.shape[0]):
                print(f"Original[{i}]:", df_original.iloc[i].values[:5])  # print first 5 values
                print(f"Reconstruction[{i}]:", df_reconstruction.iloc[i].values[:5])
                print(f"Reconstruction RE[{i}]:", df_reconstruction_re.iloc[i].values[:5])

                plt.figure(figsize=(10, 3))
                plt.plot(df_original.iloc[i], label='original', color='blue')
                plt.plot(df_reconstruction.iloc[i], label='reconstruction', color='orange', linestyle='--', marker='o', markersize=3)
                plt.plot(df_reconstruction_re.iloc[i], label='reconstruction_using_newobj', color='green')
                plt.title(f'Sequence {i}')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()


            result_w = {
                'Alpha_constraint': True,
                'C': c,
                'Lambda': lambda_,
                'MaxInnerIteration': maxInnerIter,
                'RE': test_recons_error_sidl_with_alpha
            }
            results.append(result_w)

            S_without_alpha_const, A_, Offsets_, F_obj_ = USIDL(train_X, train_y, lambda_, K, q, c, epsilon, maxIter,
                                                                maxInnerIter, runid)
            # print("Learnt shapelets without the positivity constraint for same data", S_without_alpha_const)
            # print("this is a trial2")

            # reconstruction error on test
            A_rand_init = np.random.randn(n_test, K)
            A_test_wo = A_rand_init
            Offsets_test_wo = np.random.randint(0, p - q, (n_test, K))
            A_test_wo, Offsets_test_wo, F_all = update_A_par(test_X, S_without_alpha_const, A_test_wo,
                                                         Offsets_test_wo, lambda_, maxIter,
                                                         epsilon)
            test_recons_error_sidl_without_alpha_raw, _ = unsup_obj(test_X, S_without_alpha_const, A_test_wo, Offsets_test_wo,
                                                             0)
            test_recons_error_sidl_without_alpha = test_recons_error_sidl_without_alpha_raw / n_test
            print(
                f'\n\n RECONS ERROR on test set without alpha constraint (C={c}, lambda={lambda_}) (maxInnerIter={maxInnerIter}): {test_recons_error_sidl_without_alpha}.\n\n')



            result_wo = {
                'Alpha_constraint': False,
                'C': c,
                'Lambda': lambda_,
                'MaxInnerIteration': maxInnerIter,
                'RE': test_recons_error_sidl_without_alpha
            }
            results.append(result_wo)

df = pd.DataFrame(results)
print(df)
df.to_csv("GunPoint_experiment_results.csv", index=False)

            # fig, axs = plt.subplots(2, 5, figsize=(15, 4))
            # axs = axs.flatten()
            # for i in range(10):
            #     axs[i].plot(range(38), S_with_alpha_const[i], color='blue')
            #     axs[i].set_title(f"Basis_with_alpha_const {i}")
            # plt.tight_layout()
            # plt.show()





            # fig, axs = plt.subplots(2, 5, figsize=(15, 4))
            # axs = axs.flatten()
            # for i in range(10):
            #     axs[i].plot(range(38), S_without_alpha_const[i], color='red')
            #     axs[i].set_title(f"Basis_without_alpha_const {i}")
            # plt.tight_layout()
            # plt.show()

# n_train, p = train_X.shape
# n_test, _ = test_X.shape
#
# c = 100
# epsilon = 1e-5
# maxIter = 1e3
# maxInnerIter = 5
#
# # Loop through a set of variables
# Ks = [20]  # Add more values if needed
# lambdas = [1]  # Add more values if needed
# rs = [0.25]  # Add more values if needed
#
# for K in Ks:
#     for lambda_ in lambdas:
#         A_rand_init = np.random.randn(n_test, K)
#         for r in rs:
#             q = int(np.ceil(p * r))
#             runid = f'{dataset_name}_l_{lambda_}_K_{K}_q_{q}'
#
#             # Train SIDL on training set
#             start_time = time.time()
#             S, A, Offsets, F_obj  = USIDL(train_X, train_y, lambda_, K, q, c, epsilon, maxIter, maxInnerIter, runid)
#             learn_time = time.time() - start_time
#             print(f'\n##### TRAINING TIME on TRAIN SET (K={K}, lambda={lambda_}, r={r}): {learn_time} secs.\n\n')
#





