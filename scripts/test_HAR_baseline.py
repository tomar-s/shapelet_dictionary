import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from helper.preprocess_data import load_csv, data_compiler, split_train_test_and_normalise

def main():

    # Loading data
    data = load_csv('/Users/shivanitomar/Downloads/001_labeled.csv', delimiter=',', skip_header=0, dtype=object)
    # X = np.array(load_csv('/Users/shivanitomar/Downloads/001_labeled.csv')[1:,1:-11], dtype=float) #consider all features
    # X = np.array(load_csv('/Users/shivanitomar/Downloads/001_labeled.csv')[1:,5:-7], dtype=float)  # consider only ankle (x, y ,z)
    X = np.array(load_csv('/Users/shivanitomar/Downloads/001_labeled.csv')[1:, 8], dtype=float)
    y = np.array(load_csv('/Users/shivanitomar/Downloads/001_labeled.csv')[1:, -1], dtype=object)
    y_new = np.array([label.decode('utf-8') if isinstance(label, bytes) else label for label in y])

    # splitting into train and test before data preprocessing into windows to avoid possible data leakage
    train_X, train_y, test_X, test_y = split_train_test_and_normalise(X, y_new)


    # preprocess the data stream using data compiler into windows for both train and test sets
    train_data, train_labels = data_compiler(train_X, train_y, method='slide', window=300, step=20, qty=500)
    test_data, test_labels = data_compiler(test_X, test_y, method='slide', window=300, step=20, qty=500)


     # train MLP classifier based on shapelet transformed features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data)
    X_test_scaled = scaler.transform(test_data)

    clf = MLPClassifier(hidden_layer_sizes=(100,),
                        activation='relu',
                        solver='adam',
                        alpha=0.0001,
                        batch_size='auto',
                        learning_rate='constant',
                        learning_rate_init=0.001,
                        max_iter=200,
                        random_state=42)
    clf.fit(X_train_scaled, train_labels)


    # Test on the data from the same distribution (subject/person)
    y_pred = clf.predict(X_test_scaled)
    print("Accuracy using MLP Classifier on raw data", accuracy_score(test_labels, y_pred))

   # Baseline 2: Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(train_data, train_labels)

    print("Accuracy using Random Forest Classifier on raw data", accuracy_score(test_labels, rf_clf.predict(test_data)))





































if __name__ == "__main__":
        main()