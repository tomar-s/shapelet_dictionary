import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, cohen_kappa_score
from river.ensemble import AdaBoostClassifier
from river.tree import HoeffdingTreeClassifier
# from river.utils import dict2numpy

def main1():

    np.random.seed(55)

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
    # clf = MLPClassifier(hidden_layer_sizes=(100,),
    #                     activation='relu',
    #                     solver='adam',
    #                     alpha=0.0001,
    #                     batch_size='auto',
    #                     learning_rate='constant',
    #                     learning_rate_init=0.001,
    #                     max_iter=200,
    #                     random_state=42)
    clf = AdaBoostClassifier(model=HoeffdingTreeClassifier())

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

        # For first iteration only partial_fit
        if i == 0:
            for epoch in range(10):
                for j in range(X_batch_norm.shape[0]):
                    x = dict(enumerate(X_batch_norm[j]))  # feature names become 0, 1, 2,...
                    y = y_batch[j]
                    clf.learn_one(x, y)
                # clf.partial_fit(X_batch_norm, y_batch, classes=np.unique(data_stream['label']))
            continue

        y_preds = []
        for x in X_batch_norm:
            y_pred = clf.predict_one(x)
            y_preds.append(y_pred)


        print(classification_report(y_batch, y_preds))

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
            clf.partial_fit(X_batch_norm, y_batch)


    df = pd.DataFrame(prequential_results)
    df.to_csv('/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/HAR_results/prequential_results_with_river_online.csv', index=False)


if __name__== "__main__":
    main1()
    # main2()
