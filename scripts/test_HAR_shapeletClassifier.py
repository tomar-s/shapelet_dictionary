from helper import utils, update_policy
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import \
    accuracy_score, \
    precision_recall_fscore_support, cohen_kappa_score
from helper.preprocess_data import load_csv, data_compiler, split_train_test_and_normalise
from sklearn.preprocessing import StandardScaler


def main():

    # Loading data
    data = load_csv('/Users/shivanitomar/Downloads/001_labeled.csv', delimiter=',', skip_header=0, dtype=object)
    X = np.array(load_csv('/Users/shivanitomar/Downloads/001_labeled.csv')[1:, 8], dtype=float)  # consider only ankle vm
    y = np.array(load_csv('/Users/shivanitomar/Downloads/001_labeled.csv')[1:, -1], dtype=object)
    y_new = np.array([label.decode('utf-8') if isinstance(label, bytes) else label for label in y])
    # splitting into train and test before data preprocessing into windows to avoid possible data leakage
    train_X, train_y, test_X, test_y = split_train_test_and_normalise(X, y_new)


    # preprocess the data stream using data compiler into windows for both train and test sets
    train_data, train_labels = data_compiler(train_X, train_y, method='slide', window=300, step=20, qty=500)
    test_data, test_labels = data_compiler(test_X, test_y, method='slide', window=300, step=20, qty=500)

    assert len(train_data) == len(train_labels)
    assert len(test_data) == len(test_labels)
    assert len(train_data[0]) == 300
    assert len(test_data[0]) == 300

    shapelets = {}
    for label in np.unique(train_labels):
        class_data = train_data[np.where(train_labels == label)[0]]
        class_labels = train_labels[np.where(train_labels == label)[0]]
        np.savetxt(f'{label}_train_data_for_shapelets.csv', class_data)
        print(f"Learning shapelets for label - {label}")
        S = utils.learn_shapelets(class_data, class_labels, K=5, lambdas=1, r=0.5)
        shapelets[label] = S

    # for label in (["downstairs"]):
    #     class_data = train_data[np.where(train_labels == label)[0]]
    #     class_labels = train_labels[np.where(train_labels == label)[0]]
    #     np.savetxt('downstairs_train_for_shapelets.csv', class_data)
    #     print(f"Learning shapelets for label - {label}")
    #     S = utils.learn_shapelets(class_data, class_labels, K=20, lambdas=0.1, r=0.5)
    #     shapelets[label] = S
    # # Learn sparse coding on same train data
    # r=0.5
    # K=20
    # n_train, p = class_data.shape
    # n_test, _ = class_data.shape
    # q = int(np.ceil(p * r))
    #
    # A_rand_init = np.random.randn(n_test, K)
    # A_test = A_rand_init
    # Offsets_test = np.random.randint(0, p - q, (n_test, K))
    # A_test, Offsets_test, _ = utils.update_A_par(class_data, S, A_test, Offsets_test, lambda_=0.1, maxIter = 1e-3 , epsilon = 1e-5)
    #
    # test_recons_error_sidl = utils.unsup_obj(test_data, S, A_test, Offsets_test, 0) / n_test
    # print("reconstruction error on same data as used to calculate shpelets", test_recons_error_sidl)

    # plot_shapelets_per_label(shapelets)
    # downstairs_shapelets = shapelets["downstairs"]
    # upstairs_shapelets = shapelets["upstairs"]
    # walk_mixed_shapelets = shapelets["walk_mixed"]
    # np.savetxt('downstairs_shapelets_5.csv', downstairs_shapelets)
    # np.savetxt('upstairs_shapelets_5.csv', upstairs_shapelets)
    # np.savetxt('walk_mixed_shapelets_5.csv', walk_mixed_shapelets)



    # data transformation using dtw distance between learnt shapelets and training set to train the MLP classifier
    all_shapelets = [s for arr in shapelets.values() for s in arr]
    all_shapelets_array = np.vstack(all_shapelets)
    transformed_train = utils.shapelet_transform(train_data, all_shapelets_array)
    transformed_test = utils.shapelet_transform(test_data, all_shapelets_array)


     # train MLP classifier based on shapelet transformed features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(transformed_train)
    X_test_scaled = scaler.transform(transformed_test)

    clf = MLPClassifier(hidden_layer_sizes=(100,),
                        activation='relu',
                        solver='adam',
                        alpha=0.0001,
                        batch_size='auto',
                        learning_rate='constant',
                        learning_rate_init=0.001,
                        max_iter=1000,
                        random_state=42)
    clf.fit(X_train_scaled, train_labels)


    # Test on the data from the same distribution (subject/person)
    y_pred = clf.predict(X_test_scaled)
    print("Accuracy of the trained Shapelet Classifier on same distribution test data", accuracy_score(test_labels, y_pred))
    print("Kappa score", cohen_kappa_score(test_labels, y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, y_pred, average='weighted')
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}\n")

    print("---------------------------------------------------------------------------------------")

    # Test on the data from the different distribution (subject/person)
    X_out = np.array(load_csv('/Users/shivanitomar/Downloads/002_labeled.csv')[1:, 9], dtype=float)
    y_out = np.array(load_csv('/Users/shivanitomar/Downloads/002_labeled.csv')[1:, -1], dtype=object)
    y_str = np.array([label.decode('utf-8') if isinstance(label, bytes) else label for label in y_out])
    train_X_out, train_y_out, test_X_out, test_y_out = split_train_test_and_normalise(X_out, y_str)

    # preprocess the data stream using data compiler into windows for both train and test sets
    train_data_out, train_labels_out = data_compiler(train_X_out, train_y_out, method='slide', window=300, step=20, qty=500)
    test_data_out, test_labels_out = data_compiler(test_X_out, test_y_out, method='slide', window=300, step=20, qty=500)

    transformed_train_out = utils.shapelet_transform(train_data_out, all_shapelets_array)
    transformed_test_out = utils.shapelet_transform(test_data_out, all_shapelets_array)

    X_test_scaled_out = scaler.transform(transformed_test_out)

    # Test on the data from different distribution (10th subject/person)
    y_pred_out = clf.predict(X_test_scaled_out)
    print("Accuracy of the trained Shapelet Classifier for out of distribution data", accuracy_score(test_labels_out, y_pred_out))
    print("Kappa score", cohen_kappa_score(test_labels_out, y_pred_out))

  # Inspect new shapelets for out of distribution data
    print("learning the shapelets of new distribution to update the dictionary based on different policies")
    shapelets_ood = {}
    for label in np.unique(train_labels_out):
        class_data = train_data_out[np.where(train_labels_out == label)[0]]
        class_labels = train_labels_out[np.where(train_labels_out == label)[0]]

        print(f"Learning shapelets for label - {label}")
        S = utils.learn_shapelets(class_data, class_labels, K=3, lambdas=1, r=0.5)
        shapelets_ood[label] = S

    # plot_shapelets_per_label(shapelets_ood)

#      try to see if the new distribution shapelets bring back the kappa score to previous high of 0.91
    all_shapelets_ood = [s for arr in shapelets_ood.values() for s in arr]
    all_shapelets_ood_array = np.vstack(all_shapelets_ood)
    transformed_train_out_ood = utils.shapelet_transform(train_data_out, all_shapelets_ood_array)
    transformed_test_out_ood = utils.shapelet_transform(test_data_out, all_shapelets_ood_array)

    # train MLP classifier based on shapelet transformed features
    scaler_new = StandardScaler()
    X_train_new_scaled = scaler_new.fit_transform(transformed_train_out_ood)
    X_test_new_scaled = scaler_new.transform(transformed_test_out_ood)

    clf_new = MLPClassifier(hidden_layer_sizes=(100,),
                        activation='relu',
                        solver='adam',
                        alpha=0.0001,
                        batch_size='auto',
                        learning_rate='constant',
                        learning_rate_init=0.001,
                        max_iter=1000,
                        random_state=42)
    clf_new.fit(X_train_new_scaled, train_labels_out)

    # Test after learning new shapelets from new distribution and using those features to train new classifier
    y_pred_new_ood = clf_new.predict(X_test_new_scaled)
    print("Accuracy of the trained Shapelet Classifier for out of distribution data",
          accuracy_score(test_labels_out, y_pred_new_ood))
    print("Kappa score", cohen_kappa_score(test_labels_out, y_pred_new_ood))


#     benchmarking the results for update policy to see which policy of Dictionary update yields the highest accuracy
    S_updated_closest = update_policy.combine_top_10_closest_shapelets(all_shapelets_ood, all_shapelets)
    print("the length of S_updated_closest is ", len(S_updated_closest))

    transformed_train_out_update1 = utils.shapelet_transform(train_data_out, S_updated_closest)
    transformed_test_out_update1 = utils.shapelet_transform(test_data_out, S_updated_closest)

    scaler_update1 = StandardScaler()
    X_train_new_scaled_1 = scaler_update1.fit_transform(transformed_train_out_update1)
    X_test_new_scaled_1 = scaler_update1.transform(transformed_test_out_update1)

    clf_update1 = MLPClassifier(hidden_layer_sizes=(100,),
                            activation='relu',
                            solver='adam',
                            alpha=0.0001,
                            batch_size='auto',
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            max_iter=1000,
                            random_state=42)
    clf_update1.fit(X_train_new_scaled_1, train_labels_out)

    # Test after updating the dictionary based on appropriate policy from new distribution and using those features to train new classifier
    y_pred_update1 = clf_update1.predict(X_test_new_scaled_1)
    print("Accuracy of the trained Shapelet Classifier after updating dictionary with policy II (Closest) for out of distribution data",
          accuracy_score(test_labels_out, y_pred_update1))
    print("Kappa score", cohen_kappa_score(test_labels_out, y_pred_update1))

    # test update policy III (dissimilar)

    S_updated_dissimilar = update_policy.combine_top_10_dissimilar_shapelets(all_shapelets_ood, all_shapelets)
    print("the length of S_updated_dissimilar is ", len(S_updated_closest))

    transformed_train_out_update2 = utils.shapelet_transform(train_data_out, S_updated_dissimilar)
    transformed_test_out_update2 = utils.shapelet_transform(test_data_out, S_updated_dissimilar)

    scaler_update2 = StandardScaler()
    X_train_new_scaled_2 = scaler_update2.fit_transform(transformed_train_out_update2)
    X_test_new_scaled_2 = scaler_update2.transform(transformed_test_out_update2)

    clf_update2 = MLPClassifier(hidden_layer_sizes=(100,),
                                activation='relu',
                                solver='adam',
                                alpha=0.0001,
                                batch_size='auto',
                                learning_rate='constant',
                                learning_rate_init=0.001,
                                max_iter=1000,
                                random_state=42)
    clf_update2.fit(X_train_new_scaled_2, train_labels_out)

    # Test after updating the dictionary based on appropriate policy from new distribution and using those features to train new classifier
    y_pred_update2 = clf_update2.predict(X_test_new_scaled_2)
    print("Accuracy of the trained Shapelet Classifier after updating dictionary with policy III (Dissimilar) for out of distribution data",
        accuracy_score(test_labels_out, y_pred_update2))
    print("Kappa score", cohen_kappa_score(test_labels_out, y_pred_update2))

    # test update policy I (Random)

    S_updated_random = update_policy.combine_random_shapelets(all_shapelets_ood, all_shapelets)
    print("the length of S_updated_random is ", len(S_updated_random))

    transformed_train_out_update3 = utils.shapelet_transform(train_data_out, S_updated_random)
    transformed_test_out_update3 = utils.shapelet_transform(test_data_out, S_updated_random)

    scaler_update3 = StandardScaler()
    X_train_new_scaled_3 = scaler_update3.fit_transform(transformed_train_out_update3)
    X_test_new_scaled_3 = scaler_update3.transform(transformed_test_out_update3)

    clf_update3 = MLPClassifier(hidden_layer_sizes=(100,),
                                activation='relu',
                                solver='adam',
                                alpha=0.0001,
                                batch_size='auto',
                                learning_rate='constant',
                                learning_rate_init=0.001,
                                max_iter=1000,
                                random_state=42)
    clf_update3.fit(X_train_new_scaled_3, train_labels_out)

    # Test after updating the dictionary based on appropriate policy from new distribution and using those features to train new classifier
    y_pred_update3 = clf_update3.predict(X_test_new_scaled_3)
    print("Accuracy of the trained Shapelet Classifier after updating dictionary with policy I (Random) for out of distribution data",
        accuracy_score(test_labels_out, y_pred_update3))
    print("Kappa score", cohen_kappa_score(test_labels_out, y_pred_update3))





if __name__ == "__main__":
        main()