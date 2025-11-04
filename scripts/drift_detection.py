import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from helper.utils import learn_shapelet_dict_via_sidl
from sklearn.linear_model import SGDClassifier
from helper.shapelet_transform import ShapeletTransform


def main():

    np.random.seed(42)

    # data stream simulated with concept drift
    data_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/datasets/HAR/balanced_data_stream_8_batches.csv"
    batch_size = 500
    label_col = "label"

    data_stream = pd.read_csv(data_path)
    threshold = None
    drift_detected = False
    shapelets_dict = None
    transformer = None
    all_shapelets = None    # the current set of shapelets at a given time
    prequential_results = []

    # initialise the classifier for incremental learning
    clf = SGDClassifier()


    # # Loop over streaming batches
    # for i in range(0, len(data_stream), batch_size):
    #     print("************************  Batch", i)
    #
    #     batch = data_stream.iloc[i:i + batch_size]
    #     if len(batch) < batch_size:
    #         break
    #
    #     X_batch_with_labels = batch.copy()
    #     X_batch = batch.drop(columns=[label_col]).values
    #     y_batch = batch[label_col].values
    #
    #     # try normalising window-wise before extracting shapelets and its distance features
    #     mean_batch = X_batch.mean(axis=1, keepdims=True)
    #     std_batch = X_batch.std(axis=1, keepdims=True)
    #     std_batch[std_batch == 0] = 1.0
    #     X_batch_norm = (X_batch - mean_batch) / std_batch
    #
    #     train_data_for_shapelet = np.hstack((X_batch_norm, y_batch.reshape(-1, 1)))
    #
    #     # For first iteration learn shapelets from initial batch
    #     if i == 0:
    #         # shapelets_dict = learn_shapelet_dict_via_sidl(X_batch_with_labels)
    #         shapelets_dict = learn_shapelet_dict_via_sidl(train_data_for_shapelet)
    #         all_shapelets = [s for s_list in shapelets_dict.values() for s in s_list]
    #
    #         # # saving for further analysis
    #         # save_dir = 'deep_analysis/full_pipeline_shapelets_unnormalised_batch_wip_run2/'  # No leading '/'
    #         # os.makedirs(save_dir, exist_ok=True)
    #         # np.savetxt(os.path.join(save_dir, 'downstairs_shapelets_10.csv'), shapelets_dict["downstairs"])
    #         # np.savetxt(os.path.join(save_dir, 'upstairs_shapelets_10.csv'), shapelets_dict["upstairs"])
    #         # np.savetxt(os.path.join(save_dir, 'walk_mixed_shapelets_10.csv'), shapelets_dict["walk_mixed"])
    #         # np.savetxt(os.path.join(save_dir, 'walk_sidewalk_shapelets_10.csv'), shapelets_dict["walk_sidewalk"])
    #         # np.savetxt(os.path.join(save_dir, 'walk_treadmill_shapelets_10.csv'), shapelets_dict["walk_treadmill"])
    #         # np.savetxt(os.path.join(save_dir, 'jog_treadmill_shapelets_10.csv'), shapelets_dict["jog_treadmill"])
    #
    #         transformer = ShapeletTransform(all_shapelets)
    #         # X_transformed = transformer.transform(X_batch)
    #         X_transformed = transformer.transform(X_batch_norm)
    #         continue
    #
    #     # checking drift using KS test on shapelet transformed features from batch 1 to batch 2
    #
    #     X_transformed_2 = transformer.transform(X_batch_norm)
    #     for i in range(X_transformed.shape[1]):
    #         stat, p_value = ks_2samp(X_transformed[:, i], X_transformed_2[:, i])
    #         print(f"Shapelet {i}: KS Statistic = {stat:.3f}, p-value = {p_value:.3f}")
    #         if p_value < 0.05:
    #             print(f" Drift detected on shapelet {i} (p < 0.05)")

        # checking drift using MMD classwise

    for i in range(0, len(data_stream), batch_size):
        print("************************  Batch", i)

        batch = data_stream.iloc[i:i + batch_size]
        if len(batch) < batch_size:
            break

        # X_batch_with_labels = batch.copy()
        batch_cw = batch[batch["label"]=="walk_mixed"]
        batch_cw_for_shapelets = batch_cw.copy()
        X_batch = batch_cw.drop(columns=[label_col]).values
        y_batch = batch_cw[label_col].values

        # try normalising window-wise before extracting shapelets and its distance features
        mean_batch = X_batch.mean(axis=1, keepdims=True)
        std_batch = X_batch.std(axis=1, keepdims=True)
        std_batch[std_batch == 0] = 1.0
        X_batch_norm = (X_batch - mean_batch) / std_batch

        train_data_for_shapelet = np.hstack((X_batch_norm, y_batch.reshape(-1, 1)))

        # For first iteration learn shapelets from initial batch
        if i == 0:
            # shapelets_dict = learn_shapelet_dict_via_sidl(X_batch_with_labels)
            shapelets_dict = learn_shapelet_dict_via_sidl(train_data_for_shapelet)
            all_shapelets = [s for s_list in shapelets_dict.values() for s in s_list]


            transformer = ShapeletTransform(all_shapelets)
            # X_transformed = transformer.transform(X_batch)
            X_transformed = transformer.transform(X_batch_norm)
            continue

        # checking drift using MMD test on shapelet transformed features from batch 1 to batch 2 only for downstairs class

        X_transformed_2 = transformer.transform(X_batch_norm)

        def compute_mmd(X, Y, gamma=1.0):
            K_XX = rbf_kernel(X, X, gamma=gamma)
            K_YY = rbf_kernel(Y, Y, gamma=gamma)
            K_XY = rbf_kernel(X, Y, gamma=gamma)
            mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
            return mmd

        mmd_score = compute_mmd(X_transformed, X_transformed_2)
        print(f"MMD between class downstairs in batch1 and batch2: {mmd_score:.4f}")
        continue



if __name__== "__main__":
    main()

















