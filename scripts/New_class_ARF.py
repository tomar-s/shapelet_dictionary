import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from helper import utils
from helper.utils import learn_shapelet_dict_via_sidl, normalise_batch
from sklearn.utils import shuffle
from helper.shapelet_transform import ShapeletTransform
import matplotlib.pyplot as plt
from river import forest
from river import metrics


def main1():

    subject_list = [1, 4, 8, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41]
    print(f"Using data from {len(subject_list)} subjects")
    df_list = []

    for i in subject_list:
        padded_id = str(i).zfill(3)
        df = pd.read_csv(f"/Users/shivanitomar/Downloads/{padded_id}_labeled.csv")
        df_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True)
    full_df.drop(full_df[full_df['activity'].isin(['walk_treadmill', 'walk_mixed'])].index, inplace=True)
    X_joined = full_df["ankle_vm"]
    y_joined = full_df["activity"]
    train_data, train_labels = utils.data_compiler(np.array(X_joined), np.array(y_joined), method='slide', window=100,
                                                   step=100, qty=10000)
    df_windows = pd.DataFrame(train_data)
    df_windows['label'] = train_labels
    epochs_list = utils.data_new_classes(df_windows)

    # class_label = {
    #     "walk_sidewalk": 0,
    #     "jog_treadmill": 1,
    #     "downstairs": 2,
    #     "upstairs": 3
    # }
    all_labels = df_windows['label'].unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    all_shapelets = None
    seen_classes = {"walk_sidewalk", "jog_treadmill"}
    model = forest.ARFClassifier(n_models=10, seed=32, drift_detector=None, #ADWIN(delta=0.0001),
                                 remove_poor_attrs=True, max_depth=3)
    accuracy_metric = metrics.Accuracy()
    kappa_metric = metrics.CohenKappa()
    prequential_accuracy = []
    kappa_score = []
    num_drift_detected = []
    num_warning_detected = []

    for epoch in range(len(epochs_list)):


        epoch_data = epochs_list[epoch]
        epoch += 1
        print("=" * 10 + f" EPOCH {epoch} " + "=" * 10)

        X_batch = epoch_data.drop(columns=["label"]).values
        y_batch = epoch_data["label"].values
        X_batch, y_batch = shuffle(X_batch, y_batch, random_state=42)
        y_encoded = label_encoder.transform(y_batch)

        # normalising window-wise before extracting shapelets and its distance features
        X_batch_norm = normalise_batch(X_batch)

        if all_shapelets is None:
            train_data_for_shapelet = np.hstack((X_batch_norm, y_batch.reshape(-1, 1)))
            shapelets_dict = learn_shapelet_dict_via_sidl(train_data_for_shapelet)
            all_shapelets = [s for s_list in shapelets_dict.values() for s in s_list]

        current_classes = set(epoch_data['label'].unique())
        print("Classes in this epoch and batch_size", current_classes, len(X_batch))
        new_classes = current_classes - seen_classes
        seen_classes.update(new_classes)

        if new_classes:
            cls = list(new_classes)
            cls_data = epoch_data[epoch_data["label"] == cls[0]]
            X_cls = cls_data.drop(columns=["label"]).values
            y_cls = cls_data["label"].values
            X_cls_norm = normalise_batch(X_cls)
            x_cls_for_shapelet = np.hstack((X_cls_norm, y_cls.reshape(-1, 1)))
            new_cls_shapelets_dict = learn_shapelet_dict_via_sidl(x_cls_for_shapelet)
            new_cls_shapelets = [s for s_list in new_cls_shapelets_dict.values() for s in s_list]
            all_shapelets.extend(new_cls_shapelets)

            # normalise x
            # pass to learn shapelet function
            # update dictionary with new class shapelets
            # update the transformer and then learn the ARF classifier
            #  check accuracy before and after learning ARF

        # Learn shapelets at each epoch to see how preq_acc changes
        # print("learning shapelet dictionary for every epoch")
        # train_data_for_shapelet = np.hstack((X_batch_norm, y_batch.reshape(-1, 1)))
        # shapelets_dict = learn_shapelet_dict_via_sidl(train_data_for_shapelet)
        # all_shapelets = [s for s_list in shapelets_dict.values() for s in s_list]

        print("Size of shapelet dictionary", len(all_shapelets))
        transformer = ShapeletTransform(all_shapelets)
        X_transformed = transformer.transform(X_batch_norm)
        # scaler = MinMaxScaler()
        # X_transformed = scaler.fit_transform(X_transformed)

        for (x, y) in zip(X_transformed, y_encoded):
            x_dict = {f"f{i}": val for i, val in enumerate(x)}
            y_pred = model.predict_one(x_dict)
            accuracy_metric.update(y, y_pred)
            kappa_metric.update(y, y_pred)
            model.learn_one(x_dict, y)
            # accuracy_metric.update(y, y_pred)

        prequential_accuracy.append(accuracy_metric.get())
        print("-" * 10 + f"Prequential Accuracy: {accuracy_metric.get()}" + "-" * 10)
        kappa_score.append(kappa_metric.get())
        print("-" * 10 + f"Kappa Score: {kappa_metric.get()}" + "-" * 10)

        num_drift_detected.append(model.n_drifts_detected())
        num_warning_detected.append(model.n_warnings_detected())
        print("Drift detected :", num_drift_detected)
        print("Warning detected :", num_warning_detected)
    print(prequential_accuracy)
    plt.plot(prequential_accuracy)
    # Add vertical lines for feature additions
    plt.axvline(x=4, color='green', linestyle='--', label='Epoch 6')
    plt.axvline(x=8, color='green', linestyle='--', label='Epoch 10')
    plt.title(f'Prequential Accuracy using shapelet features')
    plt.xlabel('Epoch')
    plt.ylabel('Preq_acc')
    plt.show()


if __name__ == "__main__":
    main1()