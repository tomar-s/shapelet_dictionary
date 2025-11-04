import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score
from helper import utils
from helper.utils import learn_shapelet_dict_via_sidl
from sklearn.linear_model import SGDClassifier
import pickle

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
    # clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    #               learning_rate='constant', learning_rate_init=0.001, max_iter=1000, random_state=35, verbose=True)
    clf = SGDClassifier()


    # Loop over streaming batches
    # for i in range(0, len(data_stream), batch_size):
    for i in range(0, 1000, batch_size):
        print("************************  Batch", i)

        batch = data_stream.iloc[i:i + batch_size]
        if len(batch) < batch_size:
            break

        # X_batch_with_labels = batch.copy()
        X_batch = batch.drop(columns=[label_col]).values
        y_batch = batch[label_col].values

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
            labeled_shapelets = [(label, shapelet) for label, s_list in shapelets_dict.items() for shapelet in s_list]


            # re_label = utils.re_for_threshold_labelwise(train_data_for_shapelet, shapelets_dict)

            base_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/deep_analysis/HAR_fStat/HAR_dict_p1.pkl"
            with open(base_path, 'wb') as f:
                pickle.dump(shapelets_dict, f)

            # # saving for further analysis
            # save_dir = 'deep_analysis/full_pipeline_shapelets_unnormalised_batch_wip_run2/'
            # os.makedirs(save_dir, exist_ok=True)
            # np.savetxt(os.path.join(save_dir, 'downstairs_shapelets_10.csv'), shapelets_dict["downstairs"])
            # np.savetxt(os.path.join(save_dir, 'upstairs_shapelets_10.csv'), shapelets_dict["upstairs"])
            # np.savetxt(os.path.join(save_dir, 'walk_mixed_shapelets_10.csv'), shapelets_dict["walk_mixed"])
            # np.savetxt(os.path.join(save_dir, 'walk_sidewalk_shapelets_10.csv'), shapelets_dict["walk_sidewalk"])
            # np.savetxt(os.path.join(save_dir, 'walk_treadmill_shapelets_10.csv'), shapelets_dict["walk_treadmill"])
            # np.savetxt(os.path.join(save_dir, 'jog_treadmill_shapelets_10.csv'), shapelets_dict["jog_treadmill"])

            # transformer = ShapeletTransform(all_shapelets)
            # X_transformed = transformer.transform(X_batch)
            # X_transformed = transformer.transform(X_batch_norm)
            start_time = time.time()
            re_inital_batch, A_batch = utils.re_for_threshold(X_batch_norm, all_shapelets)
            print("RE initial batch", re_inital_batch)
            print("time taken to calculate re for threshold on X_batch in secs", time.time()-start_time)
            threshold = re_inital_batch + (0.30 * re_inital_batch)

            for epoch in range(10):
                clf.partial_fit(A_batch, y_batch, classes=np.unique(data_stream['label']))
            # start_time = time.time()

            A_batch_df = pd.DataFrame(A_batch)
            A_batch_df.to_csv("/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/deep_analysis/HAR_fStat/A_batch.csv")
            continue

        # for second set of batches (each person) we only predict and partial_fit
        if i == 500 or i == 1500:

            # X_transformed = transformer.transform(X_batch_norm)
            re_inital_batch, A_batch = utils.re_for_threshold(X_batch_norm, all_shapelets)
            y_pred = clf.predict(A_batch)
            print(f"The performance on the batch {i} *************")
            print(classification_report(y_batch, y_pred))
            acc_drift = accuracy_score(y_batch, y_pred)
            f1_drift = f1_score(y_batch, y_pred, average='macro', zero_division=0)
            precision_drift = precision_score(y_batch, y_pred, average='macro', zero_division=0)
            print(f"Batch {i} — Accuracy: {acc_drift:.4f}, F1: {f1_drift:.4f}, Precision: {precision_drift:.4f}")
            for epoch in range(10):
                clf.partial_fit(A_batch, y_batch)
            continue

        #This code will only run for batch = 1000 (P2 starting batch)

        # for batch p2 first find RE using previous shapelets along with their sparse coeff matrix
        # start = time.time()
        # re_batch_old_shapelets, A_old = utils.re_for_threshold(X_batch_norm, all_shapelets)
        # print(f"Reconstruction error on batch {i} on initial shapelets", re_batch_old_shapelets)
        # print("Time taken to calculate RE ", time.time()-start)
        #
        # # for batch p3 find RE using its own shapelets along with the sparse coeff matrix
        #
        # shapelets_p2 = learn_shapelet_dict_via_sidl(train_data_for_shapelet)
        # all_shapelets_p2 = [s for s_list in shapelets_p2.values() for s in s_list]
        # labeled_shapelets_p2 = [(label, shapelet) for label, s_list in shapelets_p2.items() for shapelet in s_list]
        #
        # re_batch_new_shapelets, A_new = utils.re_for_threshold(X_batch_norm, all_shapelets_p2)
        # print(f"Reconstruction error on batch {i} with its exclusive shapelets", re_batch_new_shapelets)
        #
        # # write a function that takes A_old and A_new and finds the relative importance of shapelets based on the cumulative coeff of each shapelet\\
        # # and then update the shapelets classwise
        #
        # updated_shapelet_dict, shapelet_origin_dict = utils.combine_shapelets_w_sparse_coeff(A_old, A_new, labeled_shapelets, labeled_shapelets_p2, train_data_for_shapelet)
        # updated_shapelets = [s for s_list in updated_shapelet_dict.values() for s in s_list]
        #
        # re_batch_updated_shapelets, A_new = utils.re_for_threshold(X_batch_norm, updated_shapelets)
        # print(f"Reconstruction error on batch {i} after updating the shapelet dictionary", re_batch_updated_shapelets)

        # # all_shapelets = updated_shapelets
        # # transformer = ShapeletTransform(updated_shapelets)
        # X_transformed = transformer.transform(X_batch_norm)
        # y_pred = clf.predict(X_transformed)
        # print("The performance on the batch where drift is detected")
        # print(classification_report(y_batch, y_pred))
        # acc_drift = accuracy_score(y_batch, y_pred)
        # f1_drift = f1_score(y_batch, y_pred, average='macro', zero_division=0)
        # precision_drift = precision_score(y_batch, y_pred, average='macro', zero_division=0)
        # print(f"Batch {i} — Accuracy: {acc_drift:.4f}, F1: {f1_drift:.4f}, Precision: {precision_drift:.4f}")
        #
        # # this is where updated dictionary features are learned by the classifier
        # for epoch in range(10):
        #     clf.partial_fit(X_transformed, y_batch)

        # base_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/deep_analysis/HAR_online/HAR_dict_p4_new.pkl"
        # with open(base_path, 'wb') as f:
        #     pickle.dump(shapelets_p2, f)
        #
        #
        # base_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/deep_analysis/HAR_online/HAR_dict_p4_sparse_updated.pkl"
        # with open(base_path, 'wb') as f:
        #     pickle.dump(updated_shapelet_dict, f)
        #
        # base_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/deep_analysis/HAR_online/HAR_origin_dict_p4.pkl"
        # with open(base_path, 'wb') as f:
        #     pickle.dump(shapelet_origin_dict, f)







        # # accuracy batch 2 (old transformed shapelet dict)
        # X_transformed = transformer.transform(X_batch_norm)
        # y_pred = clf.predict(X_transformed)
        # print("The performance on the batch where drift is detected")
        # print(classification_report(y_batch, y_pred))
        # acc_drift = accuracy_score(y_batch, y_pred)
        # f1_drift = f1_score(y_batch, y_pred, average='macro', zero_division=0)
        # precision_drift = precision_score(y_batch, y_pred, average='macro', zero_division=0)
        # print(f"Batch {i} — Accuracy: {acc_drift:.4f}, F1: {f1_drift:.4f}, Precision: {precision_drift:.4f}")
        #
        # # accuracy batch 2 (after update transformed shapelet dict)
        # transformer = ShapeletTransform(updated_shapelets)
        # X_transformed = transformer.transform(X_batch_norm)
        # y_pred = clf.predict(X_transformed)
        # print("The performance on the batch where drift is detected")
        # print(classification_report(y_batch, y_pred))
        # acc_drift = accuracy_score(y_batch, y_pred)
        # f1_drift = f1_score(y_batch, y_pred, average='macro', zero_division=0)
        # precision_drift = precision_score(y_batch, y_pred, average='macro', zero_division=0)
        # print(f"Batch {i} — Accuracy: {acc_drift:.4f}, F1: {f1_drift:.4f}, Precision: {precision_drift:.4f}")
        #
        #
        #



        # #  check drift_detection criteria based on reconstruction error
        # # since we assume labelled data being available at each batch, we can use RE
        # #  Logic - if RE on current batch > threshold (Drift has occurred)
        # re_current_batch = utils.re_for_threshold(X_batch_norm, all_shapelets)
        # print(f"Reconstruction error on batch {i} and threshold is ", re_current_batch, threshold)
        # if re_current_batch > threshold:
        #     drift_detected = True
        #
        #
        # # update dictionary logic goes in this block
        # if drift_detected:
        #     # check metrics before updating the dictionary
        #     X_transformed = transformer.transform(X_batch_norm)
        #     y_pred = clf.predict(X_transformed)
        #     print("The performance on the batch where drift is detected")
        #     print(classification_report(y_batch, y_pred))
        #     acc_drift = accuracy_score(y_batch, y_pred)
        #     f1_drift = f1_score(y_batch, y_pred, average='macro', zero_division=0)
        #     precision_drift = precision_score(y_batch, y_pred, average='macro', zero_division=0)
        #     print(f"Batch {i} — Accuracy: {acc_drift:.4f}, F1: {f1_drift:.4f}, Precision: {precision_drift:.4f}")
        #
        #     prequential_results.append({
        #         "batch": i,
        #         "accuracy": acc_drift,
        #         "f1_score": f1_drift,
        #         "precision": precision_drift,
        #         "drift_detected": drift_detected
        #     })
        #
        #     # now dictionary update logic (first learn new shapelets)
        #     new_shapelets = learn_shapelet_dict_via_sidl(train_data_for_shapelet)
        #
        #     updated_shapelet_dict = combine_similar_labelwise(shapelets_dict, new_shapelets)
        #     updated_shapelets = [s for s_list in updated_shapelet_dict.values() for s in s_list]
        #
        #     # base_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/deep_analysis/HAR_online/HAR_dict_p2.pkl"
        #     # with open(base_path, 'wb') as f:
        #     #     pickle.dump(new_shapelets, f)
        #     #
        #     # base_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/deep_analysis/HAR_online/HAR_dict_combined.pkl"
        #     # with open(base_path, 'wb') as f:
        #     #     pickle.dump(updated_shapelet_dict, f)
        #
        #     all_shapelets = updated_shapelets      # updated_shapelets saved to current set of shapelets
        #
        #     # check metrics after the dictionary is updated
        #     transformer = ShapeletTransform(all_shapelets)
        #     X_transformed = transformer.transform(X_batch_norm)
        #     y_pred = clf.predict(X_transformed)
        #     print("The performance on the batch where drift is detected after the dictionary is updated")
        #     acc_drift = accuracy_score(y_batch, y_pred)
        #     f1_drift = f1_score(y_batch, y_pred, average='macro', zero_division=0)
        #     precision_drift = precision_score(y_batch, y_pred, average='macro', zero_division=0)
        #     print(f"Batch {i} — Accuracy: {acc_drift:.4f}, F1: {f1_drift:.4f}, Precision: {precision_drift:.4f}")
        #
        #     # update threshold based on new distribution
        #     re_current = utils.re_for_threshold(X_batch_norm, all_shapelets)
        #     threshold = re_current + (0.3 * re_current)
        #
        #     # fit the model on new distribution
        #     for epoch in range(50):
        #         clf.partial_fit(X_transformed, y_batch)
        #     drift_detected = False
        #     continue
        #
        # X_transformed = transformer.transform(X_batch_norm)
        # y_pred = clf.predict(X_transformed)
        #
        # print(classification_report(y_batch, y_pred))
        #
        # acc = accuracy_score(y_batch, y_pred)
        # f1 = f1_score(y_batch, y_pred, average='macro', zero_division=0)
        # precision = precision_score(y_batch, y_pred, average='macro', zero_division=0)
        # print(f"Batch {i} — Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}")
        #
        # prequential_results.append({
        #     "batch": i,
        #     "accuracy": acc,
        #     "f1_score": f1,
        #     "precision": precision,
        #     "drift_detected": drift_detected
        # })
        # df = pd.DataFrame(prequential_results)
        # df.to_csv('/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/HAR_results/prequential_results.csv', index=False)
        #
        # for epoch in range(50):
        #     clf.partial_fit(X_transformed, y_batch)


if __name__== "__main__":
    main()

















