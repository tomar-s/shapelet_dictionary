import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, cohen_kappa_score
from helper import utils
from helper.utils import learn_shapelet_dict_via_sidl
from sklearn.linear_model import SGDClassifier
from helper.shapelet_transform import ShapeletTransform


def main1():

    np.random.seed(55)

    # data stream simulated with concept drift
    # data_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/datasets/HAR/balanced_data_stream_20_batches.csv"
    data_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/datasets/HAR/balanced_data_stream_20_batches.csv"
    batch_size = 500
    label_col = "label"

    data_stream = pd.read_csv(data_path)
    threshold = None
    drift_detected = False
    transformer = None
    all_shapelets = None    # the current set of shapelets at a given time
    prequential_results = []

    # initialise the classifier for incremental learning
    clf = SGDClassifier()


    # Loop over streaming batches
    for i in range(0, len(data_stream), batch_size):

        batch_num = i // batch_size + 1

        print("*************  Batch", batch_num)

        batch = data_stream.iloc[i:i + batch_size]
        if len(batch) < batch_size:
            break

        # X_batch_with_labels = batch.copy()
        X_batch = batch.drop(columns=[label_col]).values
        y_batch = batch[label_col].values

        # normalising window-wise before extracting shapelets and its distance features
        mean_batch = X_batch.mean(axis=1, keepdims=True)
        std_batch = X_batch.std(axis=1, keepdims=True)
        std_batch[std_batch == 0] = 1.0
        X_batch_norm = (X_batch - mean_batch) / std_batch

        train_data_for_shapelet = np.hstack((X_batch_norm, y_batch.reshape(-1, 1)))

        # For first iteration learn shapelets from initial batch
        if i == 0:

            shapelets_dict = learn_shapelet_dict_via_sidl(train_data_for_shapelet)
            all_shapelets = [s for s_list in shapelets_dict.values() for s in s_list]

            transformer = ShapeletTransform(all_shapelets)
            X_transformed = transformer.transform(X_batch_norm)

            for epoch in range(10):
                clf.partial_fit(X_transformed, y_batch, classes=np.unique(data_stream['label']))

            start_time = time.time()
            re_inital_batch, _ = utils.re_for_threshold(X_batch_norm, all_shapelets)
            print("Time taken to calculate re for threshold on X_batch in secs", time.time()-start_time)
            threshold = re_inital_batch + (0.30 * re_inital_batch)
            continue


        #  check drift_detection criteria based on reconstruction error
        #  Logic - if RE on current batch > threshold (Drift has occurred)
        re_current_batch, _ = utils.re_for_threshold(X_batch_norm, all_shapelets)
        print(f"Reconstruction error on current batch and threshold is ", re_current_batch, threshold)
        if re_current_batch > threshold:
            drift_detected = True


        # update dictionary logic goes in this block
        if drift_detected:
            # check metrics before updating the dictionary
            X_transformed = transformer.transform(X_batch_norm)
            y_pred = clf.predict(X_transformed)
            print("The performance on the batch where drift is detected")
            print(classification_report(y_batch, y_pred))
            acc_drift = accuracy_score(y_batch, y_pred)
            f1_drift = f1_score(y_batch, y_pred, average='macro', zero_division=0)
            precision_drift = precision_score(y_batch, y_pred, average='macro', zero_division=0)
            kappa_score = cohen_kappa_score(y_batch, y_pred)
            print(f"Batch {batch_num} — Accuracy: {acc_drift:.4f}, F1: {f1_drift:.4f}, Precision: {precision_drift:.4f}, Kappa: {kappa_score: 4f}")

            prequential_results.append({
                "batch": batch_num,
                "accuracy": acc_drift,
                "f1_score": f1_drift,
                "precision": precision_drift,
                "Kappa": kappa_score,
                "drift_detected": drift_detected
            })

            # now dictionary update logic (first learn new shapelets)
            new_shapelets = learn_shapelet_dict_via_sidl(train_data_for_shapelet)

            # updated_shapelet_dict = combine_similar_labelwise(shapelets_dict, new_shapelets)
            updated_shapelets = [s for s_list in new_shapelets.values() for s in s_list]

            all_shapelets = updated_shapelets      # updated_shapelets saved to current set of shapelets

            # check metrics after the dictionary is updated
            transformer = ShapeletTransform(all_shapelets)
            X_transformed = transformer.transform(X_batch_norm)
            y_pred = clf.predict(X_transformed)
            print("The performance on the batch where drift is detected after the dictionary is updated")
            acc_drift = accuracy_score(y_batch, y_pred)
            f1_drift = f1_score(y_batch, y_pred, average='macro', zero_division=0)
            precision_drift = precision_score(y_batch, y_pred, average='macro', zero_division=0)
            kappa_score = cohen_kappa_score(y_batch, y_pred)
            print(f"Batch {batch_num} — Accuracy: {acc_drift:.4f}, F1: {f1_drift:.4f}, Precision: {precision_drift:.4f}, Kappa: {kappa_score: 4f}")

            # update threshold based on new distribution
            re_current, _ = utils.re_for_threshold(X_batch_norm, all_shapelets)
            threshold = re_current + (0.3 * re_current)
            print(f"Reconstruction error on current batch and threshold is ", re_current, threshold)

            # fit the model on new distribution
            for epoch in range(10):
                clf.partial_fit(X_transformed, y_batch)

            drift_detected = False
            continue

        X_transformed = transformer.transform(X_batch_norm)
        y_pred = clf.predict(X_transformed)

        print(classification_report(y_batch, y_pred))

        acc = accuracy_score(y_batch, y_pred)
        f1 = f1_score(y_batch, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_batch, y_pred, average='macro', zero_division=0)
        kappa_score = cohen_kappa_score(y_batch, y_pred)
        print(f"Batch {batch_num} — Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Kappa: {kappa_score: 4f}")

        prequential_results.append({
            "batch": batch_num,
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision,
            "Kappa": kappa_score,
            "drift_detected": drift_detected
        })

        for epoch in range(10):
            clf.partial_fit(X_transformed, y_batch)


    df = pd.DataFrame(prequential_results)
    df.to_csv('/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/HAR_results/prequential_results_with_update.csv', index=False)


def main2():

    np.random.seed(485)

    # data stream simulated with concept drift
    data_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/datasets/HAR/balanced_data_stream_20_batches.csv"
    batch_size = 500
    label_col = "label"

    data_stream = pd.read_csv(data_path)
    threshold = None
    drift_detected = False
    transformer = None
    all_shapelets = None    # the current set of shapelets at a given time
    prequential_results = []

    # initialise the classifier for incremental learning
    clf = SGDClassifier()

    # Loop over streaming batches
    for i in range(0, len(data_stream), batch_size):

        batch_num = i // batch_size + 1

        print("*************  Batch", batch_num)

        batch = data_stream.iloc[i:i + batch_size]
        if len(batch) < batch_size:
            break

        # X_batch_with_labels = batch.copy()
        X_batch = batch.drop(columns=[label_col]).values
        y_batch = batch[label_col].values

        # normalising window-wise before extracting shapelets and its distance features
        mean_batch = X_batch.mean(axis=1, keepdims=True)
        std_batch = X_batch.std(axis=1, keepdims=True)
        std_batch[std_batch == 0] = 1.0
        X_batch_norm = (X_batch - mean_batch) / std_batch

        train_data_for_shapelet = np.hstack((X_batch_norm, y_batch.reshape(-1, 1)))

        # For first iteration learn shapelets from initial batch
        if i == 0:

            shapelets_dict = learn_shapelet_dict_via_sidl(train_data_for_shapelet)
            all_shapelets = [s for s_list in shapelets_dict.values() for s in s_list]
            labeled_shapelets = [(label, shapelet) for label, s_list in shapelets_dict.items() for shapelet in s_list]

            transformer = ShapeletTransform(all_shapelets)
            X_transformed = transformer.transform(X_batch_norm)

            for epoch in range(10):
                clf.partial_fit(X_transformed, y_batch, classes=np.unique(data_stream['label']))

            # start_time = time.time()
            # re_inital_batch, _ = utils.re_for_threshold(X_batch_norm, all_shapelets)
            # print("Time taken to calculate re for threshold on X_batch in secs", time.time()-start_time)
            # threshold = re_inital_batch + (0.30 * re_inital_batch)
            continue


        #  check drift_detection criteria based on reconstruction error
        #  Logic - if RE on current batch > threshold (Drift has occurred)
        # re_current_batch, _ = utils.re_for_threshold(X_batch_norm, all_shapelets)
        # print(f"Reconstruction error on current batch and threshold is ", re_current_batch, threshold)
        if batch_num % 2 != 0:
            drift_detected = True


        # update dictionary logic goes in this block
        if drift_detected:

            start = time.time()
            re_batch_old_shapelets, A_old = utils.re_for_threshold(X_batch_norm, all_shapelets)
            print(f"Reconstruction error on batch {batch_num} on initial shapelets", re_batch_old_shapelets)
            print("Time taken to calculate RE ", time.time()-start)

            # for batch p3 find RE using its own shapelets along with the sparse coeff matrix

            shapelets_p2 = learn_shapelet_dict_via_sidl(train_data_for_shapelet)
            all_shapelets_p2 = [s for s_list in shapelets_p2.values() for s in s_list]
            labeled_shapelets_p2 = [(label, shapelet) for label, s_list in shapelets_p2.items() for shapelet in s_list]

            re_batch_new_shapelets, A_new = utils.re_for_threshold(X_batch_norm, all_shapelets_p2)
            print(f"Reconstruction error on batch {batch_num} with its exclusive shapelets", re_batch_new_shapelets)

            updated_shapelet_dict, shapelet_origin_dict = utils.combine_shapelets_w_sparse_coeff(A_old, A_new,
                                                                                                 labeled_shapelets,
                                                                                                 labeled_shapelets_p2,
                                                                                                 train_data_for_shapelet)
            updated_shapelets = [s for s_list in updated_shapelet_dict.values() for s in s_list]
            all_shapelets = updated_shapelets

            # check metrics before updating the dictionary
            X_transformed = transformer.transform(X_batch_norm)
            y_pred = clf.predict(X_transformed)
            print("The performance on the batch where drift is detected")
            print(classification_report(y_batch, y_pred))
            acc_drift = accuracy_score(y_batch, y_pred)
            f1_drift = f1_score(y_batch, y_pred, average='macro', zero_division=0)
            precision_drift = precision_score(y_batch, y_pred, average='macro', zero_division=0)
            kappa_score = cohen_kappa_score(y_batch, y_pred)
            print(f"Batch {batch_num} — Accuracy: {acc_drift:.4f}, F1: {f1_drift:.4f}, Precision: {precision_drift:.4f}, Kappa: {kappa_score: 4f}")

            prequential_results.append({
                "batch": batch_num,
                "accuracy": acc_drift,
                "f1_score": f1_drift,
                "precision": precision_drift,
                "Kappa": kappa_score,
                "drift_detected": drift_detected
            })

            # # now dictionary update logic (first learn new shapelets)
            # new_shapelets = learn_shapelet_dict_via_sidl(train_data_for_shapelet)
            #
            # # updated_shapelet_dict = combine_similar_labelwise(shapelets_dict, new_shapelets)
            # updated_shapelets = [s for s_list in new_shapelets.values() for s in s_list]
            #
            # all_shapelets = updated_shapelets      # updated_shapelets saved to current set of shapelets

            # check metrics after the dictionary is updated
            transformer = ShapeletTransform(all_shapelets)
            X_transformed = transformer.transform(X_batch_norm)
            y_pred = clf.predict(X_transformed)
            print("The performance on the batch where drift is detected after the dictionary is updated")
            acc_drift = accuracy_score(y_batch, y_pred)
            f1_drift = f1_score(y_batch, y_pred, average='macro', zero_division=0)
            precision_drift = precision_score(y_batch, y_pred, average='macro', zero_division=0)
            kappa_score = cohen_kappa_score(y_batch, y_pred)
            print(f"Batch {batch_num} — Accuracy: {acc_drift:.4f}, F1: {f1_drift:.4f}, Precision: {precision_drift:.4f}, Kappa: {kappa_score: 4f}")

            # # update threshold based on new distribution
            # re_current, _ = utils.re_for_threshold(X_batch_norm, all_shapelets)
            # threshold = re_current + (0.3 * re_current)
            # print(f"Reconstruction error on current batch and threshold is ", re_current, threshold)

            # fit the model on new distribution
            for epoch in range(10):
                clf.partial_fit(X_transformed, y_batch)

            drift_detected = False
            continue

        X_transformed = transformer.transform(X_batch_norm)
        y_pred = clf.predict(X_transformed)

        print(classification_report(y_batch, y_pred))

        acc = accuracy_score(y_batch, y_pred)
        f1 = f1_score(y_batch, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_batch, y_pred, average='macro', zero_division=0)
        kappa_score = cohen_kappa_score(y_batch, y_pred)
        print(f"Batch {batch_num} — Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Kappa: {kappa_score: 4f}")

        prequential_results.append({
            "batch": batch_num,
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision,
            "Kappa": kappa_score,
            "drift_detected": drift_detected
        })

        for epoch in range(10):
            clf.partial_fit(X_transformed, y_batch)


    df = pd.DataFrame(prequential_results)
    df.to_csv('/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/HAR_results/prequential_results_with_update_run4.csv', index=False)




if __name__== "__main__":
    # main1()
    main2()

















