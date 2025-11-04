import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import \
    accuracy_score, \
    confusion_matrix, precision_recall_fscore_support, cohen_kappa_score


def main_w_shapeletFeatures():

    dataset_name = 'sine_RW_mode_20000_no_windows'
    drift_data = pd.read_csv('../../datasets/sine_RW_from_cpnn/sine_rw10_mode5_1conf_20000_no_windows.csv')

    batch_size = 1000
    # window_size = 20

    # Initial batch for shapelet learning
    train_data = drift_data[0:1000]
    test_data = drift_data[1000:2000]

    X_train = train_data.drop(columns=['target', 'task']).values
    y_train = train_data['target'].values
    X_test = test_data.drop(columns=['target', 'task']).values
    y_test = test_data['target'].values


    # ax = sns.countplot(x=y_train, color='#3bb143')
    # ax.set(
    #     title='Target distribution is balanced? Yes!',
    #     xlabel='Label'
    # )
    # ax.axhline(y=400, color='#0b6623', linestyle='--')

    rem_data = drift_data[2000:]



    # Get unique classes (needed for partial_fit)
    classes = np.unique(y_train)

    # Training MLP classifier incrementally
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=[128, 32,],
        activation='relu',
        solver='adam',  # High-performance stochastic gradient-based optimizer
        alpha=5e-3,  # Strength of the L2 regularization term (added to the loss)
        learning_rate_init=5e-4,  #
        shuffle=False,
        random_state=42,
        verbose=0,
    )


    mlp_clf.partial_fit(X_train, y_train, classes = classes)  # Train on initial 500 samples

    # predict on the test samples from concept 1(same concept it was initially trained on)
    y_pred = mlp_clf.predict(X_test)

    # Compute performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    print("Accuracy on initial batch", accuracy)
    print("confusion matrix", conf )
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"Cohen's Kappa Score: {kappa:.4f}")

    # Performance tracking
    performance_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        "kappa_score": []
    }

    for i in range(0, len(rem_data), batch_size):

        # if len(test_batch) < batch_size // 2 or len(train_batch) < batch_size // 2:
        #     print("Skipping batch due to insufficient data")
        #     continue

        batch_last_idx = min(i + batch_size, len(rem_data))
        X_batch = rem_data[i:batch_last_idx].drop(columns=['target', 'task']).values
        y_batch = rem_data[i:batch_last_idx]['target'].values

        # Prequential evaluation
        y_pred = mlp_clf.predict(X_batch)

        # Compute performance metrics
        accuracy_start = accuracy_score(y_batch, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_batch, y_pred, average='weighted', zero_division=1
        )
        kappa_start = cohen_kappa_score(y_batch, y_pred)
        print(f"Cohen's Kappa Score: {kappa_start:.4f}")

        # Store metrics
        performance_metrics['accuracy'].append(accuracy_start)
        performance_metrics['precision'].append(precision)
        performance_metrics['recall'].append(recall)
        performance_metrics['f1_score'].append(f1)
        performance_metrics['kappa_score'].append(kappa_start)

        print("At the start of new concept")
        print(f"Batch {i // batch_size}:")
        print(f"Accuracy: {accuracy_start:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}\n")
        print(f"Cohen's Kappa Score: {kappa_start:.4f}")

        # train after test (prequential training)
        mlp_clf.partial_fit(X_batch, y_batch)

        #At concept end
        print("At the end of new concept")
        y_pred = mlp_clf.predict(X_batch)

        kappa_end = cohen_kappa_score(y_batch, y_pred)
        accuracy_end = accuracy_score(y_batch, y_pred)
        print(f"Cohen's Kappa Score: {kappa_end:.4f}")
        print(f"Accuracy: {accuracy_end:.4f}")


    print("\nFinal Performance:")
    print(performance_metrics)

if __name__ == "__main__":
    # main_wo_shapeletFeatures()
    main_w_shapeletFeatures()
