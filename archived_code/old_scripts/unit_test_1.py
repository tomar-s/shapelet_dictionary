import time
import pandas as pd
import numpy as np
from archived_code import utils
from sklearn.metrics import accuracy_score, classification_report
# from utils import learn_shapelet_dict_via_sidl, learn_shapelets_alpha_const
from sklearn.neural_network import MLPClassifier
from shapelet_transform import ShapeletTransform
from helper.preprocess_data import split_train_test, load_csv, data_compiler
from sklearn.preprocessing import StandardScaler

# This file executes unit_test 1 where we take only 1 batch split into train and test to see if learning shapelets (unsupervised or classwise)
# which scenarios is more accurate and the interpretability of learnt shapelets.

def main():
    np.random.seed(42)

    X = np.array(load_csv('/Users/shivanitomar/Downloads/001_labeled.csv')[1:, 8], dtype=float)  # consider only ankle vm
    y = np.array(load_csv('/Users/shivanitomar/Downloads/001_labeled.csv')[1:, -1], dtype=object)
    y_new = np.array([label.decode('utf-8') if isinstance(label, bytes) else label for label in y])

    # splitting into train and test before data preprocessing into windows to avoid possible data leakage
    train_X, train_y, test_X, test_y = split_train_test(X, y_new)

    # preprocess the data stream using data compiler into windows for both train and test sets
    train_data, train_labels = data_compiler(train_X, train_y, method='slide', window=300, step=20, qty=500)
    test_data, test_labels = data_compiler(test_X, test_y, method='slide', window=300, step=20, qty=500)

    assert len(train_data) == len(train_labels)
    assert len(test_data) == len(test_labels)
    assert len(train_data[0]) == 300
    assert len(test_data[0]) == 300


#  Scenario 1 : Learn 30 shapelets unsupervised (K=30, lambdas=1, r=0.25) without any normalisation till now
    start_time = time.time()
    print("Scenario 1")
    shapelets, A, Offsets, F_obj = utils.learn_shapelets_alpha_const(train_data, train_labels, K=60, lambdas=1, r=0.25)
    # shapelets = np.loadtxt('/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/scripts/deep_analysis/unit_tests_shapelets_without_labels_subject1/30_shapelets_without_labels_seed42.csv')
    print("Time taken for convergence of SIDL when finding shapelet dictionary without labels in seconds", time.time() - start_time )
    # save_dir = 'deep_analysis/unit_tests_shapelets_without_labels_subject1/'
    # os.makedirs(save_dir, exist_ok=True)
    # np.savetxt(os.path.join(save_dir, '30_shapelets_without_labels_seed42.csv'), shapelets)

#     transformed dataset based on these 30 unsupervised shapelets
    start_time = time.time()
    transformer = ShapeletTransform(shapelets)
    X_transformed_train = transformer.transform(train_data)
    X_transformed_test = transformer.transform(test_data)
    print("Time taken for shapelet transformation in secs", time.time() - start_time)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_transformed_train)
    X_test_scaled = scaler.transform(X_transformed_test)

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
    print("Accuracy of the trained Shapelet Classifier when shapelet transformation is carried out using unsupervised shapelets",
          accuracy_score(test_labels, y_pred))
    print(classification_report(test_labels, y_pred))


# Scenario 2 : Learn 30 shapelets (K =5, lambdas = 1, r=0.25) 5 for each label (supervised) without any normalisation till now
    print("Scenario 2")
    train_df = pd.DataFrame(train_data)
    train_df['label'] = train_labels
    shapelets_dict = utils.learn_shapelet_dict_via_sidl(train_df)
    all_shapelets = [s for arr in shapelets_dict.values() for s in arr]
    all_shapelets_array = np.vstack(all_shapelets)

    # # saving for further analysis
    # save_dir = 'deep_analysis/unit_tests_shapelets_with_labels_subject1/'  # No leading '/'
    # os.makedirs(save_dir, exist_ok=True)
    # np.savetxt(os.path.join(save_dir, 'downstairs_shapelets_5.csv'), shapelets_dict["downstairs"])
    # np.savetxt(os.path.join(save_dir, 'upstairs_shapelets_5.csv'), shapelets_dict["upstairs"])
    # np.savetxt(os.path.join(save_dir, 'walk_mixed_shapelets_5.csv'), shapelets_dict["walk_mixed"])
    # np.savetxt(os.path.join(save_dir, 'walk_sidewalk_shapelets_5.csv'), shapelets_dict["walk_sidewalk"])
    # np.savetxt(os.path.join(save_dir, 'walk_treadmill_shapelets_5.csv'), shapelets_dict["walk_treadmill"])
    # np.savetxt(os.path.join(save_dir, 'jog_treadmill_shapelets_5.csv'), shapelets_dict["jog_treadmill"])

    #transformed dataset based on these 30 labelwise shapelets
    start_time = time.time()
    transformer_sc2 = ShapeletTransform(all_shapelets_array)
    X_transformed_train = transformer_sc2.transform(train_data)
    X_transformed_test = transformer_sc2.transform(test_data)
    print("Time taken for shapelet transformation in secs", time.time()-start_time )

    scaler_sc2 = StandardScaler()
    X_train_scaled = scaler_sc2.fit_transform(X_transformed_train)
    X_test_scaled = scaler_sc2.transform(X_transformed_test)

    clf_sc2 = MLPClassifier(hidden_layer_sizes=(100,),
                        activation='relu',
                        solver='adam',
                        alpha=0.0001,
                        batch_size='auto',
                        learning_rate='constant',
                        learning_rate_init=0.001,
                        max_iter=1000,
                        random_state=42)
    clf_sc2.fit(X_train_scaled, train_labels)

    # Test on the data from the same distribution (subject/person)
    y_pred = clf_sc2.predict(X_test_scaled)
    print("Accuracy of the trained Shapelet Classifier when shapelet transformation is carried out using supervised shapelets",
          accuracy_score(test_labels, y_pred))
    print(classification_report(test_labels, y_pred))




if __name__ == "__main__":
        main()