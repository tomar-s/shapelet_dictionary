from archived_code import utils
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix


class IncrementalSVMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 learning_rate='optimal',
                 loss='hinge',
                 penalty='l2',
                 alpha=0.0001,
                 max_iter=1000,
                 random_state=None):
        """
        Incremental SVM Classifier using Stochastic Gradient Descent

        Parameters:
        -----------
        learning_rate : str, default='optimal'
            Learning rate schedule
        loss : str, default='hinge'
            Loss function
        penalty : str, default='l2'
            Regularization penalty
        alpha : float, default=0.0001
            Regularization strength
        max_iter : int, default=1000
            Maximum number of iterations
        random_state : int, optional
            Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

        # Initialize components
        self.classifier = None
        self.scaler = StandardScaler()

        # Performance tracking
        self.performance_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }

    def fit(self, X, y):
        """
        Initial model training

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target values

        Returns:
        --------
        self : object
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize SGD classifier (SVM variant)
        self.classifier = SGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

        # Train initial model
        self.classifier.fit(X_scaled, y)

        return self

    def predict(self, X):
        """
        Predict class labels

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input features

        Returns:
        --------
        predictions : array-like of shape (n_samples,)
            Predicted class labels
        """
        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X):
        """
        Predict class probabilities

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input features

        Returns:
        --------
        probabilities : array-like of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        return self.classifier.predict_proba(X_scaled)

    def partial_fit(self, X, y, classes=None):
        """
        Incremental learning method

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target values
        classes : array-like, optional
            List of all possible classes

        Returns:
        --------
        self : object
        """
        # Scale features
        X_scaled = self.scaler.partial_fit(X).transform(X)

        # Initialize classifier if not already done
        if self.classifier is None:
            self.classifier = SGDClassifier(
                loss=self.loss,
                penalty=self.penalty,
                alpha=self.alpha,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                random_state=self.random_state
            )

        # Partial fit with new data
        self.classifier.partial_fit(X_scaled, y, classes)

        return self


def prequential_evaluation(X, y, S, test_size=200, batch_size=100):
    """
    Prequential (Progressive Validation) Evaluation with Shapelet Transformation.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Input features
    y : array-like of shape (n_samples,)
        Target values
    S : array-like
        Learned shapelets used for transformation
    test_size : int, default=200
        Number of initial samples used for testing
    batch_size : int, default=100
        Size of incremental learning batches

    Returns:
    --------
    performance_metrics : dict
        Tracking of model performance across iterations
    """

    # First `test_size` samples for initial testing (no training yet)
    X_test = X[:test_size]
    y_test = y[:test_size]

    # Remaining data for training
    X_train = X[test_size:]
    y_train = y[test_size:]

    # Transform test data using learned shapelets
    shapelet_features_test = utils.shapelet_transform(X_test, S)

    # Initialize incremental SVM
    inc_svm = IncrementalSVMClassifier()

    # Performance tracking
    performance_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }

    # Get unique classes (needed for partial_fit)
    classes = np.unique(y_train)

    # Prequential evaluation
    for i in range(0, len(X_train), batch_size):
        # Get current batch
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Transform batch using shapelets
        shapelet_features_batch = utils.shapelet_transform(X_batch, S)

        # 1. **Test on unseen data before training**
        if inc_svm.classifier is not None:
            y_pred = inc_svm.predict(shapelet_features_test)

            # Compute performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )

            # Store metrics
            performance_metrics['accuracy'].append(accuracy)
            performance_metrics['precision'].append(precision)
            performance_metrics['recall'].append(recall)
            performance_metrics['f1_score'].append(f1)

            print(f"Batch {i // batch_size}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}\n")

        # 2. **Train on the current batch**
        if i == 0:  # Pass `classes` only in the first call
            inc_svm.partial_fit(shapelet_features_batch, y_batch, classes=classes)
        else:
            inc_svm.partial_fit(shapelet_features_batch, y_batch)

    return performance_metrics


def main_wo_shapeletFeatures():

    dataset_name = 'sine_RW_mode_4000'
    drift_data = pd.read_csv('../../datasets/sine_RW_from_cpnn/sine_rw10_mode5_1conf_4000.csv')

    batch_size = 981

    # Initial batch for training without shapelet based features
    train_data = drift_data[0:500]
    test_data = drift_data[500:951]

    train_X = train_data.to_numpy()[:, :50]
    train_y = train_data.to_numpy()[:, 50:51].ravel()
    test_X = test_data.to_numpy()[:, :50]
    test_y = test_data.to_numpy()[:, 50:51].ravel()

    rem_data = drift_data[951:]
    # Get unique classes (needed for partial_fit)
    classes = np.unique(train_y)

    # Train Incremental SVM
    inc_svm = SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=0.0001,  # Learning rate parameter
        max_iter=1000,
        random_state=87,  # FIX RANDOM SEED
        shuffle=False  # Ensure batch order is fixed
    )
    inc_svm.partial_fit(train_X, train_y, classes = classes)  # Train on initial 500 samples

    # predict on the test samples from concept 1(same concept it was initially trained on)
    y_pred = inc_svm.predict(test_X)

    # Compute performance metrics
    accuracy = accuracy_score(test_y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y, y_pred, average='weighted'
    )
    print("Accuracy on initial concept",accuracy)

    # Performance tracking
    performance_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }

    data = []

    for i in range(0, len(rem_data), batch_size):
        # divide the current batch to train and test for prequential evaluation
        e_s = i + batch_size // 2
        test_batch = rem_data.to_numpy()[i:e_s, :50]
        train_batch = rem_data.to_numpy()[e_s: e_s + batch_size // 2, :50]

        if len(test_batch) < batch_size // 2 or len(train_batch) < batch_size // 2:
            print("Skipping batch due to insufficient data")
            continue

        print("Length of test batch :", len(test_batch))
        print("Length of train batch :", len(train_batch))

        y_test = rem_data.to_numpy()[i:e_s, 50:51].ravel()
        y_train = rem_data.to_numpy()[e_s: e_s + batch_size // 2, 50:51].ravel()

        print(inc_svm.coef_)
        data.append({
            # 'c1': inc_svm.coef_.flatten()[0],
            # 'c2': inc_svm.coef_.flatten()[1],
            **{f'c{j + 1}': inc_svm.coef_.flatten()[j] for j in range(50)},
            'mod_sgd': np.mean(inc_svm.predict(test_batch) == y_test),
            'i': i + train_batch.shape[0]
        })
        y_pred = inc_svm.predict(test_batch)

        # Compute performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=1
        )

        # Store metrics
        performance_metrics['accuracy'].append(accuracy)
        performance_metrics['precision'].append(precision)
        performance_metrics['recall'].append(recall)
        performance_metrics['f1_score'].append(f1)

        print(f"Batch {i // batch_size}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}\n")

        # train after test (prequential training)
        inc_svm.partial_fit(train_batch, y_train)

    print("\nFinal Performance:")
    print(performance_metrics)



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

    # Train Incremental SVM
    inc_svm = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=0.0001,  # Learning rate parameter
        max_iter=1000,
        random_state=87,  # FIX RANDOM SEED
        shuffle=False  # Ensure batch order is fixed
    )
    inc_svm.partial_fit(shapelet_features_train, train_y, classes = classes)  # Train on initial 500 samples

    # predict on the test samples from concept 1(same concept it was initially trained on)
    y_pred = inc_svm.predict(shapelet_features_test)

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

        print(inc_svm.coef_)
        y_pred = inc_svm.predict(shapelet_features_test)

        # Compute performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=1
        )
        kappa = cohen_kappa_score(y_test, y_pred)
        print(f"Cohen's Kappa Score: {kappa:.4f}")

        # Store metrics
        performance_metrics['accuracy'].append(accuracy)
        performance_metrics['precision'].append(precision)
        performance_metrics['recall'].append(recall)
        performance_metrics['f1_score'].append(f1)
        performance_metrics['kappa_score'].append(kappa)

        print(f"Batch {i // batch_size}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}\n")
        print(f"Cohen's Kappa Score: {kappa:.4f}")

        # train after test (prequential training)
        inc_svm.partial_fit(shapelet_features_train, y_train)

    print("\nFinal Performance:")
    print(performance_metrics)

if __name__ == "__main__":
    # main_wo_shapeletFeatures()
    main_w_shapeletFeatures()
