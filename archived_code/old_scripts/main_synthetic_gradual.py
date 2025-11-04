from archived_code import utils, update_policy
import numpy as np
import pandas as pd
import time

# Set random seed
np.random.seed(10)

# For synthetic gradual drift data
dataset_name = 'synthetic_gradual_3000_seed10'
drift_data = pd.read_csv('../../datasets/concept_drift_synthetic/gradual_using_AR_3000_seed10.csv')


train_data = drift_data[:1000]
topk_test_data = drift_data[1000:2000]
topk_test_X  = topk_test_data.to_numpy()[:, :100]
test_data = drift_data[2000:]
train_X = train_data.to_numpy()[:, :100]
train_y = train_data.to_numpy()[:, 100]
test_X  = test_data.to_numpy()[:, :100]
test_y = test_data.to_numpy()[:, 100]
n_train, p = train_X.shape
n_test, _ = test_X.shape

c = 100
epsilon = 1e-5
maxIter = 1e3
maxInnerIter = 5

# Loop through a set of variables
Ks = [20]  # Add more values if needed
lambdas = [0.1, 1]  # Add more values if needed
rs = [0.1, 0.25, 0.5]  # Add more values if needed

for K in Ks:
    for lambda_ in lambdas:
        A_rand_init = np.random.randn(n_test, K)
        for r in rs:
            q = int(np.ceil(p * r))
            runid = f'{dataset_name}_l_{lambda_}_K_{K}_q_{q}'

            # Train SIDL on training set
            start_time = time.time()
            S, A, Offsets, F_obj  = utils.USIDL(train_X, train_y, lambda_, K, q, c, epsilon, maxIter, maxInnerIter, runid)
            learn_time = time.time() - start_time
            print(f'\n##### TRAINING TIME on TRAIN SET (K={K}, lambda={lambda_}, r={r}): {learn_time} secs.\n\n')

            # Learn sparse coding on test set with dictionary learned from training set
            A_test = A_rand_init
            Offsets_test = np.random.randint(0, p-q, (n_test, K))
            start_time = time.time()
            A_test, Offsets_test, F_all_1 = utils.update_A_par(test_X, S, A_test, Offsets_test, lambda_, maxIter, epsilon)
            fit_time_1 = time.time() - start_time
            print(f"\n**** TIME TO SPARSE CODING MATRIX ON TEST SET: {fit_time_1} secs.")

            # Get reconstruction for SIDL using training set shapelets
            start_time = time.time()
            test_recons_error_sidl = utils.unsup_obj(test_X, S, A_test, Offsets_test, 0) / n_test
            fit_time_2 = time.time() - start_time
            print(f'\n\n RECONS ERROR after GRADUAL DRIFT without updating shapelets (K={K}, lambda={lambda_}) SIDL (r={r}): {test_recons_error_sidl} TIME TAKEN : {fit_time_2} secs.\n\n')

            utils.visualize_results(test_X, S, A_test, Offsets_test, sample_idx=55, plot_dir=f'final_plots/plots_Syn_gradual_shapelets/{dataset_name}')

            # shapelet ranking based on same distribution (1000 points) as the one used to learn shapelets
            # Learn sparse coding on topK_test_set with dictionary learned from training set
            A_topK_test = A_rand_init
            Offsets_topK_test = np.random.randint(0, p - q, (n_test, K))
            start_time = time.time()
            A_topK_test, _, _ = utils.update_A_par(topk_test_X, S, A_topK_test, Offsets_topK_test, lambda_, maxIter,
                                                   epsilon)
            fit_time_1 = time.time() - start_time



            sorted_A_test, sorted_indices, overall_rank, shapelet_top_k_rank, mean_ranks, top_k_counts = utils.rank_shapelets(A_topK_test)

            #
            # print("Overall Shapelet Ranking (lower rank = more important):")
            # for i, shapelet in enumerate(overall_rank):
            #     print(f"Rank {i + 1}: Shapelet {shapelet} (Mean Rank: {mean_ranks[shapelet]:.2f})")
            #
            # print("\nShapelet Frequency in Top-10 Across Samples:")
            # for i, shapelet in enumerate(shapelet_top_k_rank):
            #     print(f"Rank {i + 1}: Shapelet {shapelet} (Top-10 Count: {top_k_counts[shapelet]})")

            # Learn new shapelets after drift has occurred (Gradual Drift)
            new_train_x = test_X[:700]
            new_train_y = test_y[:700]
            new_test_x = test_X[700:]
            new_test_y = test_y[700:]
            n_train_new, p_new = new_train_x.shape
            n_test_new, _ = new_test_x.shape

            # Train SIDL on new training set (after distribution changes)
            K_new = K - 10
            start_time = time.time()
            S_after_drift, A_after_drift, Offsets_after_drift, F_obj = utils.USIDL(new_train_x, new_train_y, lambda_, K_new, q,
                                                                                   c, epsilon, maxIter, maxInnerIter, runid)
            learn_time = time.time() - start_time
            print(f'\n##### TRAINING TIME on new TRAIN SET after drift occurred (K={K}, lambda={lambda_}, r={r}): {learn_time} secs.\n\n')

            # Testing all 4 update policies : Random/Closest/Dissimilar/TopK

            # Policy I : Discard random shapelets from previous distribution to update shapelet dictionary
            S_updated_random = update_policy.combine_random_shapelets(S_after_drift, S)
            print("the length of S_updated_random is ", len(S_updated_random))

            # Learn sparse coding on held out test set with updated dictionary using policy I
            A_random = np.random.randn(n_test_new, K)
            Offsets_random = np.random.randint(0, p - q, (n_test_new, K))
            start_time = time.time()
            A_random, Offsets_random, F_all_ = utils.update_A_par(new_test_x, S_updated_random, A_random, Offsets_random,
                                                                  lambda_, maxIter, epsilon)
            fit_time_1 = time.time() - start_time
            print(f"\n**** TIME TO SPARSE CODING MATRIX ON held out TEST SET after drift: {fit_time_1} secs.")

            # Get reconstruction for SIDL using updated shapelets using policy I
            start_time = time.time()
            test_recons_error_sidl_random = utils.unsup_obj(new_test_x, S_updated_random, A_random, Offsets_random, 0) / n_test_new
            fit_time_2 = time.time() - start_time
            print(f'\n\n##### RECONS ERROR on held out TEST SET with updated shapelets using policy I (Random) (K={K}, lambda={lambda_}) SIDL (r={r}): {test_recons_error_sidl_random} TIME TAKEN : {fit_time_2} secs.\n\n')
            utils.visualize_results(new_test_x, S_updated_random, A_random, Offsets_random, sample_idx=200,
                                    plot_dir="../../final_plots/synthetic_gradual/updated_shapelets_policyI_(Random)")

            # Policy II : Keep closest 10 shapelets from previous distribution to update shapelet dictionary
            S_updated_closest = update_policy.combine_top_10_closest_shapelets(S_after_drift, S)
            print("the length of S_updated_closest is ", len(S_updated_closest))

            # Learn sparse coding on held out test set with updated dictionary using policy II
            A_closest = np.random.randn(n_test_new, K)
            Offsets_closest = np.random.randint(0, p - q, (n_test_new, K))
            start_time = time.time()
            A_closest, Offsets_closest, F_all_closest = utils.update_A_par(new_test_x, S_updated_closest, A_closest, Offsets_closest, lambda_, maxIter, epsilon)
            fit_time_1 = time.time() - start_time
            print(f"\n**** TIME TO SPARSE CODING MATRIX ON held out TEST SET after drift: {fit_time_1} secs.")

            # Get reconstruction for SIDL using updated shapelets using policy II
            start_time = time.time()
            test_recons_error_sidl_closest = utils.unsup_obj(new_test_x, S_updated_closest, A_closest, Offsets_closest, 0) / n_test_new
            fit_time_2 = time.time() - start_time
            print(f'\n\n##### RECONS ERROR on held out TEST SET with updated shapelets using policy II (Closest) (K={K}, lambda={lambda_}) SIDL (r={r}): {test_recons_error_sidl_closest} TIME TAKEN : {fit_time_2} secs.\n\n')
            utils.visualize_results(new_test_x, S_updated_closest, A_closest, Offsets_closest, sample_idx=200,
                                    plot_dir="../../final_plots/synthetic_gradual/updated_shapelets_policyII_(Closest)")


            # Policy III : Keep 10 most dissimilar shapelets from previous distribution to update shapelet dictionary
            S_updated_dissimilar = update_policy.combine_top_10_dissimilar_shapelets(S_after_drift, S)
            print("the length of S_updated_dissimilar is ", len(S_updated_dissimilar))

            # Learn sparse coding on held out test set with updated dictionary using policy II
            A_dissimilar = np.random.randn(n_test_new, K)
            Offsets_dissimilar = np.random.randint(0, p - q, (n_test_new, K))
            start_time = time.time()
            A_dissimilar, Offsets_dissimilar, F_all_dissimilar = utils.update_A_par(new_test_x, S_updated_dissimilar, A_dissimilar,
                                                                                    Offsets_dissimilar, lambda_, maxIter, epsilon)
            fit_time_1 = time.time() - start_time
            print(f"\n**** TIME TO SPARSE CODING MATRIX ON held out TEST SET after drift: {fit_time_1} secs.")

            # Get reconstruction for SIDL using updated shapelets using policy II
            start_time = time.time()
            test_recons_error_sidl_dissimilar = utils.unsup_obj(new_test_x, S_updated_dissimilar, A_dissimilar, Offsets_dissimilar, 0) / n_test_new
            fit_time_2 = time.time() - start_time
            print(f'\n\n##### RECONS ERROR on held out TEST SET with updated shapelets using policy III (Dissimilar) (K={K}, lambda={lambda_}) SIDL (r={r}): {test_recons_error_sidl_dissimilar} TIME TAKEN : {fit_time_2} secs.\n\n')
            utils.visualize_results(new_test_x, S_updated_dissimilar, A_dissimilar, Offsets_dissimilar, sample_idx=200,
                                    plot_dir="../../final_plots/synthetic_gradual/updated_shapelets_policyIII_(Dissimilar)")

            # Policy IV : Keep topK (10) shapelets from previous distribution to update shapelet dictionary
            S_updated_topK = update_policy.combine_topK_previous(S_after_drift, S, overall_rank)
            print("the length of S_updated_topK is ", len(S_updated_topK))

            # Learn sparse coding on held out test set with updated dictionary using policy II
            A_topK = np.random.randn(n_test_new, K)
            Offsets_topK = np.random.randint(0, p - q, (n_test_new, K))
            start_time = time.time()
            A_topK, Offsets_topK, F_all_topK = utils.update_A_par(new_test_x, S_updated_topK,
                                                                  A_topK,
                                                                  Offsets_topK, lambda_,
                                                                  maxIter, epsilon)
            fit_time_1 = time.time() - start_time
            print(f"\n**** TIME TO SPARSE CODING MATRIX ON held out TEST SET after drift: {fit_time_1} secs.")

            # Get reconstruction for SIDL using updated shapelets using policy II
            start_time = time.time()
            test_recons_error_sidl_topK = utils.unsup_obj(new_test_x, S_updated_topK, A_topK,
                                                          Offsets_topK, 0) / n_test_new
            fit_time_2 = time.time() - start_time
            print(
                f'\n\n##### RECONS ERROR on held out TEST SET with updated shapelets using policy IV (TopK) (K={K}, lambda={lambda_}) SIDL (r={r}): {test_recons_error_sidl_topK} TIME TAKEN : {fit_time_2} secs.\n\n')
            utils.visualize_results(new_test_x, S_updated_topK, A_topK, Offsets_topK, sample_idx=200,
                                    plot_dir="../../final_plots/synthetic_gradual/updated_shapelets_policyIV_(TopK)")



