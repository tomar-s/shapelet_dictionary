import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from helper import utils
from helper.utils import learn_shapelet_dict_via_sidl
from helper.shapelet_transform import ShapeletTransform


def main():

    seed = 33
    np.random.seed(seed)
    data_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/datasets/RAN/mob_pattern.csv"
    data = pd.read_csv(data_path)

    train_data = data[:600]
    test_data = data[600:]

    train_x, train_y = train_data.drop(columns=["label"]).values, train_data["label"].values
    test_x, test_y = test_data.drop(columns=["label"]).values, test_data["label"].values


    # perform cross validation on K, q, lambda

    data_0 = data[data["label"] == 0]
    data_1 = data[data["label"] == 1]
    data_2 = data[data["label"] == 2]
    class_list = [data_0, data_1, data_2]

    for i, cls in enumerate(class_list):

        print(f"Cross validation for class {i}")

        c = 100
        epsilon = 1e-6
        maxIter = 2e3
        maxInnerIter = 5

        class_0 = cls.drop(columns=["label"]).values
        split_idx = int(len(class_0) * 0.8)
        train = class_0[:split_idx]
        test = class_0[split_idx:]
        n, p = train.shape
        n_test = test.shape[0]

        ks = [5]  # 5, 10, 15, 20, 25, 30
        rs =  [0.5]  # 0.1, 0.2, 0.3, 0.4, 0.5
        lambdas = [1]  # 0.1, 1, 10
        for k in ks:
            for lambdas_ in lambdas:
                A_rand_init = np.random.rand(n_test, k)
                for r in rs:
                    q = int(np.ceil(p * r))
                    S, A, Offsets, F_obj = utils.USIDL_with_alpha_const(train, y = None, lambda_=lambdas_, K=k, q=q, c=c,
                                                                        epsilon=epsilon,
                                                                        maxIter=maxIter, maxInnerIter=maxInnerIter, runid=0)
                    # for i in range(len(S)):
                    #     plt.figure()
                    #     plt.plot(S[i])
                    #     plt.show()

                    shapelet_dict = {f'shapelet_{i}': S[i] for i in range(S.shape[0])}
                    # learn sparse coding on test set using dictionary from training
                    A_test = A_rand_init
                    Offsets_test = np.random.randint(0, p - q, (n_test, k))

                    A_test, Offsets_test, F_all_1 = utils.update_A_par_with_alpha_const(test, S, A_test, Offsets_test,
                                                                                        lambdas_, maxIter,
                                                                                        epsilon)

                    re, reconst_x = utils.reconstruction_err(test, S, A_test, Offsets_test)
                    # for i in range(3):
                    #     plt.figure()
                    #     plt.plot(reconst_x[i])
                    #     plt.show()

                    print(f'reconstruction error on held out test set for K = {k}, r = {r}, lambda = {lambdas_}  is ', re)

    #  Learning shapelets offline on train set
    shapelets_dict = learn_shapelet_dict_via_sidl(train_data)
    all_shapelets = [s for s_list in shapelets_dict.values() for s in s_list]
    k = len(all_shapelets)

    n_test = len(test_data)
    n_train = len(train_data)
    p = len(train_x[0])
    q = len(all_shapelets[0])
    print("length of shapelets for this run", q)

    A_train_init = np.random.rand(n_train, k)
    offset_train = np.random.randint(0, p - q, (n_train, k))
    A_test_init = np.random.rand(n_test, k)
    Offsets_test = np.random.randint(0, p - q, (n_test, k))
    A_test, _, _ = utils.update_A_par_with_alpha_const(test_x, np.array(all_shapelets), A_test_init, Offsets_test,
                                                       0.1, 2e3,
                                                       1e-6)
    A_train, _, _ = utils.update_A_par_with_alpha_const(train_x, np.array(all_shapelets), A_train_init, offset_train,
                                                        0.1, 2e3,
                                                        1e-6)
    transformer = ShapeletTransform(all_shapelets)
    X_transformed_train = transformer.transform(train_x)
    X_transformed_test = transformer.transform(test_x)

    clf1 = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                  learning_rate='constant', learning_rate_init=0.001, max_iter=1000, random_state=seed, verbose=False)
    clf1.fit(X_transformed_train, train_y)
    y_pred = clf1.predict(X_transformed_test)
    print(f"Classification Accuracy using MLP classifier with shapelet features: {accuracy_score(test_y, y_pred) * 100:.2f}%")
    print(classification_report(test_y, y_pred))
    f1 = f1_score(test_y, y_pred, average='macro', zero_division=0)
    print("f1 score", f1)
    precision = precision_score(test_y, y_pred, average='macro', zero_division=0)
    print("precision", precision)

    #  Training using sparse coeff matrix as features
    clf_sm = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.001, max_iter=1000, random_state=seed,
                         verbose=False)
    clf_sm.fit(A_train, train_y)
    y_pred_sm = clf_sm.predict(A_test)
    print(f"Classification Accuracy using MLP classifier with sparse matrix as features: {accuracy_score(test_y, y_pred_sm) * 100:.2f}%")



    clf2 = SVC(kernel="linear", C=1.0, gamma="scale", max_iter=1000, random_state=seed)
    clf2.fit(X_transformed_train, train_y)
    y_pred = clf2.predict(X_transformed_test)
    print(f"Classification Accuracy using SVM classifier with shapelet features: {accuracy_score(test_y, y_pred) * 100:.2f}%")
    print(classification_report(test_y, y_pred))
    f1 = f1_score(test_y, y_pred, average='macro', zero_division=0)
    print("f1 score", f1)
    precision = precision_score(test_y, y_pred, average='macro', zero_division=0)
    print("precision", precision)

    clf2_sm = SVC(kernel="linear", C=1.0, gamma="scale", max_iter=1000, random_state=seed)
    clf2_sm.fit(A_train, train_y)
    y_pred = clf2_sm.predict(A_test)
    print(f"Classification Accuracy using SVM classifier with sparse matrix as features: {accuracy_score(test_y, y_pred) * 100:.2f}%")




if __name__ == "__main__":
    main()