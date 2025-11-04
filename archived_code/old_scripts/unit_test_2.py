import os
import pandas as pd
import numpy as np
from archived_code import utils
# from utils import learn_shapelet_dict_via_sidl, learn_shapelets_alpha_const
from helper.preprocess_data import split_train_test, load_csv, data_compiler


# Goal of this unit test is to see the difference in classwise shapelets from Person 1 to Person 2 (change of distribution scenario)
# to ascertain the distance similarity to use in update policies.


def main():

    # learning shapelets for Distribution 1 (Person 1)
    np.random.seed(42)

    X_p1 = np.array(load_csv('/Users/shivanitomar/Downloads/001_labeled.csv')[1:, 8], dtype=float)  # consider only ankle vm
    y_p1 = np.array(load_csv('/Users/shivanitomar/Downloads/001_labeled.csv')[1:, -1], dtype=object)
    y_new_p1 = np.array([label.decode('utf-8') if isinstance(label, bytes) else label for label in y_p1])

    # splitting into train and test before data preprocessing into windows to avoid possible data leakage
    train_X, train_y, test_X, test_y = split_train_test(X_p1, y_new_p1)

    # preprocess the data stream using data compiler into windows for both train and test sets
    train_data, train_labels = data_compiler(train_X, train_y, method='slide', window=300, step=20, qty=500)
    test_data, test_labels = data_compiler(test_X, test_y, method='slide', window=300, step=20, qty=500)

    assert len(train_data) == len(train_labels)
    assert len(test_data) == len(test_labels)
    assert len(train_data[0]) == 300
    assert len(test_data[0]) == 300

    train_df = pd.DataFrame(train_data)
    train_df['label'] = train_labels
    shapelets_dict = utils.learn_shapelet_dict_via_sidl(train_df)
    # all_shapelets = [s for arr in shapelets_dict.values() for s in arr]
    # all_shapelets_array = np.vstack(all_shapelets)

    # saving for further analysis
    save_dir = '../../scripts/deep_analysis/unit_test_2/person1_shapelets_10'  # No leading '/'
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(os.path.join(save_dir, 'downstairs_shapelets_10.csv'), shapelets_dict["downstairs"])
    np.savetxt(os.path.join(save_dir, 'upstairs_shapelets_10.csv'), shapelets_dict["upstairs"])
    np.savetxt(os.path.join(save_dir, 'walk_mixed_shapelets_10.csv'), shapelets_dict["walk_mixed"])
    np.savetxt(os.path.join(save_dir, 'walk_sidewalk_shapelets_10.csv'), shapelets_dict["walk_sidewalk"])
    np.savetxt(os.path.join(save_dir, 'walk_treadmill_shapelets_10.csv'), shapelets_dict["walk_treadmill"])
    np.savetxt(os.path.join(save_dir, 'jog_treadmill_shapelets_10.csv'), shapelets_dict["jog_treadmill"])

    # learning shapelets for Distribution 2 (Person 2)
    X_p2 = np.array(load_csv('/Users/shivanitomar/Downloads/004_labeled.csv')[1:, 8],dtype=float)  # consider only ankle vm
    y_p2 = np.array(load_csv('/Users/shivanitomar/Downloads/004_labeled.csv')[1:, -1], dtype=object)
    y_new_p2 = np.array([label.decode('utf-8') if isinstance(label, bytes) else label for label in y_p2])

    train_X_p2, train_y_p2, test_X_p2, test_y_p2 = split_train_test(X_p2, y_new_p2)

    # preprocess the data stream using data compiler into windows for both train and test sets
    train_data_p2, train_labels_p2 = data_compiler(train_X_p2, train_y_p2, method='slide', window=300, step=20, qty=500)
    test_data, test_labels = data_compiler(test_X_p2, test_y_p2, method='slide', window=300, step=20, qty=500)

    train_df_p2 = pd.DataFrame(train_data_p2)
    train_df_p2['label'] = train_labels_p2
    shapelets_dict = utils.learn_shapelet_dict_via_sidl(train_df_p2)
    # all_shapelets = [s for arr in shapelets_dict.values() for s in arr]
    # all_shapelets_array = np.vstack(all_shapelets)

    # saving for further analysis
    save_dir = '../../scripts/deep_analysis/unit_test_2/person2_shapelets_10'  # No leading '/'
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(os.path.join(save_dir, 'downstairs_shapelets_10.csv'), shapelets_dict["downstairs"])
    np.savetxt(os.path.join(save_dir, 'upstairs_shapelets_10.csv'), shapelets_dict["upstairs"])
    np.savetxt(os.path.join(save_dir, 'walk_mixed_shapelets_10.csv'), shapelets_dict["walk_mixed"])
    np.savetxt(os.path.join(save_dir, 'walk_sidewalk_shapelets_10.csv'), shapelets_dict["walk_sidewalk"])
    np.savetxt(os.path.join(save_dir, 'walk_treadmill_shapelets_10.csv'), shapelets_dict["walk_treadmill"])
    np.savetxt(os.path.join(save_dir, 'jog_treadmill_shapelets_10.csv'), shapelets_dict["jog_treadmill"])

if __name__ == "__main__":
    main()