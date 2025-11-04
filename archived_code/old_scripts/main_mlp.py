from archived_code import utils
import time
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import \
    accuracy_score, \
    confusion_matrix, precision_recall_fscore_support, cohen_kappa_score


def main_w_shapeletFeatures():

    dataset_name = 'sine_RW_mode_4000_win20stride1'
    drift_data = pd.read_csv(
        '../../datasets/sine_RW_from_cpnn/sine_rw10_mode5_1conf_4000_OveralppingWindows20stride1.csv')

    batch_size = 981
    window_size = 20

    # Initial batch for shapelet learning
    train_data = drift_data[0:490]
    test_data = drift_data[490:981]

    train_X = train_data.to_numpy()[:, :window_size]
    train_y = train_data.to_numpy()[:, window_size:window_size+1].ravel()
    test_X  = test_data.to_numpy()[:, :window_size]
    test_y = test_data.to_numpy()[:, window_size:window_size+1].ravel()
    n_train, p = train_X.shape
    n_test, _ = test_X.shape

    ax = sns.countplot(x=train_y, color='#3bb143')
    ax.set(
        title='Target distribution is balanced? Yes!',
        xlabel='Label'
    )
    ax.axhline(y=400, color='#0b6623', linestyle='--')

    rem_data = drift_data[981:]


    c = 100
    epsilon = 1e-5
    maxIter = 1e3
    maxInnerIter = 5

    # Loop through a set of variables
    Ks = [20]  # Add more values if needed
    lambdas = [0.1]  # Add more values if needed
    rs = [0.5]  # Add more values if needed

    for K in Ks:
        for lambda_ in lambdas:
            # A_rand_init = np.random.randn(n_test, K)
            for r in rs:
                q = int(np.ceil(p * r))
                runid = f'{dataset_name}_l_{lambda_}_K_{K}_q_{q}'

                # Train SIDL on training set
                start_time = time.time()
                S, A, Offsets, F_obj  = utils.USIDL(train_X, train_y, lambda_, K, q, c, epsilon, maxIter, maxInnerIter, runid)
                print("Time taken(in secs) to learn shapelets from initial batch", time.time() - start_time)
                # for i in range (len(S)):
                #     plt.plot(S[i])

    shapelet_features_train = utils.shapelet_transform(train_X, S)
    shapelet_features_test = utils.shapelet_transform(test_X, S)

    # Get unique classes (needed for partial_fit)
    classes = np.unique(train_y)

    # Training MLP classifier incrementally
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=[100, ],
        activation='relu',
        solver='adam',  # High-performance stochastic gradient-based optimizer
        alpha=5e-3,  # Strength of the L2 regularization term (added to the loss)
        learning_rate_init=5e-4,  #
        shuffle=False,
        random_state=42,
        verbose=0,
    )


    mlp_clf.partial_fit(shapelet_features_train, train_y, classes = classes)  # Train on initial 500 samples

    # predict on the test samples from concept 1(same concept it was initially trained on)
    y_pred = mlp_clf.predict(shapelet_features_test)

    # Compute performance metrics
    accuracy = accuracy_score(test_y, y_pred)
    conf = confusion_matrix(test_y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y, y_pred, average='weighted'
    )
    print("Accuracy on initial concept", accuracy)
    print("confusion matrix", conf )
    kappa = cohen_kappa_score(test_y, y_pred)
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
        # divide the current batch to train and test for prequential evaluation
        e_s = i + batch_size // 2
        test_batch = rem_data.to_numpy()[i:e_s, :window_size]
        train_batch = rem_data.to_numpy()[e_s: e_s + batch_size // 2, :window_size]

        if len(test_batch) < batch_size // 2 or len(train_batch) < batch_size // 2:
            print("Skipping batch due to insufficient data")
            continue

        print("Length of test batch :", len(test_batch))
        print("Length of train batch :", len(train_batch))

        y_test = rem_data.to_numpy()[i:e_s, window_size:window_size+1].ravel()
        y_train = rem_data.to_numpy()[e_s: e_s + batch_size // 2, window_size:window_size+1].ravel()

        # Transform batch using shapelets
        shapelet_features_train = utils.shapelet_transform(train_batch, S)
        shapelet_features_test = utils.shapelet_transform(test_batch, S)

        y_pred = mlp_clf.predict(shapelet_features_test)

        # Compute performance metrics
        accuracy_start = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=1
        )
        kappa_start = cohen_kappa_score(y_test, y_pred)
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
        mlp_clf.partial_fit(shapelet_features_train, y_train)

        #At concept end
        print("At the end of new concept")
        y_pred = mlp_clf.predict(shapelet_features_test)

        kappa_end = cohen_kappa_score(y_test, y_pred)
        accuracy_end = accuracy_score(y_test, y_pred)
        print(f"Cohen's Kappa Score: {kappa_end:.4f}")
        print(f"Accuracy: {accuracy_end:.4f}")


    print("\nFinal Performance:")
    print(performance_metrics)

if __name__ == "__main__":
    # main_wo_shapeletFeatures()
    main_w_shapeletFeatures()
