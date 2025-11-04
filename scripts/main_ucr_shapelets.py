from helper import utils
import numpy as np
import time
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set random seed
np.random.seed(10)

# This is for loading GunPoint data for replicating the SIDL paper results
dataset_name = 'GunPoint'
# data = loadmat(f'/datasets/preliminary/{dataset_name}/GP_data_array.mat')
data = loadmat("/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/datasets/preliminary/GunPoint/GP_data_array.mat")
train_table = data["train_table"]
test_table = data["test_table"]

test_X  = test_table[:, 1:]
test_y  = test_table[:, 0]

# separate data corresponding to each class to create a separate sub-dictionary for that class
train_class_1 = train_table[train_table[:, 0] == 1]
train_class_2 = train_table[train_table[:, 0] == 2]

arr_list = [train_class_1, train_class_2]
all_shapelets = []

for arr in arr_list:
    train_X = arr[:, 1:]
    train_y = arr[:, 0]   # Select the first column (index 0)
    n_train, p = train_X.shape

    c = 100
    epsilon = 1e-5
    maxIter = 1e3
    maxInnerIter = 5

    # Loop through a set of variables
    Ks = [10]  # Add more values if needed
    lambdas = [1]  # Add more values if needed
    rs = [0.25]  # Add more values if needed

    for K in Ks:
        for lambda_ in lambdas:
            # A_rand_init = np.random.randn(n_test, K)
            # A_rand_init_2 = np.random.randn(n_test_2, K)
            for r in rs:
                q = int(np.ceil(p * r))
                runid = f'{dataset_name}_l_{lambda_}_K_{K}_q_{q}'

                # Train SIDL on training set
                start_time = time.time()
                S, A, Offsets, F_obj = utils.USIDL_with_alpha_const(train_X, train_y, lambda_, K, q, c, epsilon, maxIter, maxInnerIter, runid)
                learn_time = time.time() - start_time
                print(f'\n##### TRAINING TIME on TRAIN SET (K={K}, lambda={lambda_}, r={r}): {learn_time} secs.\n\n')
                all_shapelets.append(S)

all_shapelets_array = np.vstack(all_shapelets)
train_X = train_table[:, 1:]
train_y = train_table[:,0]

shapelet_features_train = utils.shapelet_transform(train_X, all_shapelets_array)
shapelet_features_test = utils.shapelet_transform(test_X, all_shapelets_array)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(shapelet_features_train, train_y)

# Evaluate model
y_pred = clf.predict(shapelet_features_test)
print(f"Classification Accuracy using RF classifier: {accuracy_score(test_y, y_pred) * 100:.2f}%")

svm_model = SVC(kernel="linear", C=1.0, gamma="scale", random_state=42)
svm_model.fit(shapelet_features_train, train_y)
y_pred = svm_model.predict(shapelet_features_test)
print(f"Classification Accuracy using SVM using shapelet transformed features: {accuracy_score(test_y, y_pred) * 100:.2f}%")


svm_model = SVC(kernel="linear", C=1.0, gamma="scale", random_state=42)
svm_model.fit(train_X, train_y)
y_pred = svm_model.predict(test_X)
print(f"Classification Accuracy using SVM using raw features: {accuracy_score(test_y, y_pred) * 100:.2f}%")


# Accuracy test after using only sparse coding(A) as features to the classifier
train_X = train_table[:, 1:]
train_y = train_table[:,0]
test_X  = test_table[:, 1:]
test_y  = test_table[:, 0]
n_test, _ = test_X.shape

c = 100
epsilon = 1e-5
maxIter = 1e3
maxInnerIter = 5

# Loop through a set of variables
Ks = [20]  # Add more values if needed
lambdas = [1]  # Add more values if needed
rs = [0.25]  # Add more values if needed

for K in Ks:
    for lambda_ in lambdas:
        A_rand_init = np.random.randn(n_test, K)
        # A_rand_init_2 = np.random.randn(n_test_2, K)
        for r in rs:
            q = int(np.ceil(p * r))
            runid = f'{dataset_name}_l_{lambda_}_K_{K}_q_{q}'

            # Train SIDL on training set
            start_time = time.time()
            S, A_train, Offsets, F_obj = utils.USIDL_with_alpha_const(train_X, train_y, lambda_, K, q, c, epsilon, maxIter, maxInnerIter, runid)
            learn_time = time.time() - start_time
            print(f'\n##### TRAINING TIME on TRAIN SET (K={K}, lambda={lambda_}, r={r}): {learn_time} secs.\n\n')

           # learn sprase coding on test set using dictionary from training
            A_test = A_rand_init
            Offsets_test = np.random.randint(0, p - q, (n_test, K))
            start_time = time.time()
            A_test, Offsets_test, F_all_1 = utils.update_A_par_with_alpha_const(test_X, S, A_test, Offsets_test, lambda_, maxIter,
                                                                                epsilon)
            fit_time_1 = time.time() - start_time
            print(f"\n**** TIME TO SPARSE CODING MATRIX ON TEST SET: {fit_time_1} secs.")

            svm_model_A = SVC(kernel="linear", C=1.0, gamma="scale", random_state=42)
            svm_model_A.fit(A_train, train_y)
            y_pred_A = svm_model_A.predict(A_test)
            print(f"Classification Accuracy using SVM using sparse coding as features: {accuracy_score(test_y, y_pred_A) * 100:.2f}%")

            clf_A = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_A.fit(A_train, train_y)
            # Evaluate model
            rf_y_pred_A = clf_A.predict(A_test)
            print(f"Classification Accuracy using RF classifier: {accuracy_score(test_y, rf_y_pred_A) * 100:.2f}%")




