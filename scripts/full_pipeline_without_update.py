import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score
from helper.utils import learn_shapelet_dict_via_sidl
from sklearn.neural_network import MLPClassifier
from helper.shapelet_transform import ShapeletTransform


def main():

    np.random.seed(42)

    data_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/datasets/HAR/balanced_data_stream_8_batches.csv"
    batch_size = 500
    label_col = "label"


    data_stream = pd.read_csv(data_path)

    prequential_results = []
    all_shapelets = None
    transformer = None
    # classifier = SGDClassifier()
    classifier = MLPClassifier()

    # Loop over streaming batches
    for i in range(0, len(data_stream), batch_size):
        batch = data_stream.iloc[i:i+batch_size]

        if len(batch) < batch_size:
            break

        X_batch_with_labels = batch.copy()
        X_batch = batch.drop(columns=[label_col]).values
        y_batch = batch[label_col].values

        print(f"\n--- Prequential Batch {i // batch_size + 1} ---")
        print("Batch size", batch_size)

        if i == 0:
            #  Learning shapelets only from the first batch (No updation of SD)
            shapelets_dict = learn_shapelet_dict_via_sidl(X_batch_with_labels)
            all_shapelets = [s for s_list in shapelets_dict.values() for s in s_list]
            transformer = ShapeletTransform(all_shapelets)
            X_transformed = transformer.transform(X_batch)
            classifier.partial_fit(X_transformed, y_batch, classes=np.unique(data_stream['label']))
            continue

        X_transformed = transformer.transform(X_batch)
        y_pred = classifier.predict(X_transformed)
        print(classification_report(y_batch, y_pred))

        acc = accuracy_score(y_batch, y_pred)
        f1 = f1_score(y_batch, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_batch, y_pred, average='macro', zero_division=0)
        print(f"Batch {i} — Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}")

        prequential_results.append({
            "batch": i,
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision
        })
        classifier.partial_fit(X_transformed, y_batch)


if __name__ == "__main__":
    main()