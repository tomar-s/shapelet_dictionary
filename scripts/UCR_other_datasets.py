import numpy as np
import pandas as pd
import time
from helper import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import os


def read_dataset(dataset_name, root_dir):

    df_train = pd.read_csv(root_dir + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)
    df_test = pd.read_csv(root_dir + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)


    y_train = df_train.values[:, 0]
    y_test = df_test.values[:, 0]

    x_train = df_train.drop(columns=[0])
    x_test = df_test.drop(columns=[0])

    x_train.columns = range(x_train.shape[1])
    x_test.columns = range(x_test.shape[1])

    x_train = x_train.values
    x_test = x_test.values

    # znorm
    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    return x_train, y_train, x_test, y_test, nb_classes

root_path = "/Users/shivanitomar/Documents/Implementations/Deep_TSC_fawaz/dl-4-tsc/archives/UCRArchive_2018"
dataset_name = "ECG5000"


def data_classwise_for_shapelets(x_train, y_train):

    features_df = pd.DataFrame(x_train)
    label_df = pd.DataFrame(y_train)
    label_df.columns = ['label']
    x_train_df = pd.concat([features_df, label_df], axis=1)

    classwise_data = {}

    for label in np.unique(y_train):
        class_data = x_train_df[y_train == label]
        class_data.drop(columns="label", inplace=True)
        class_data = np.array(class_data)
        classwise_data[label]= class_data

    return classwise_data


train_X, train_y, test_X, test_y, n_classes = read_dataset(dataset_name, root_path)
print(train_X.shape)
print(test_X.shape)
label_wise_dict = data_classwise_for_shapelets(train_X, train_y)

all_shapelets = []
shapelet_dict = {}

for key, value in label_wise_dict.items():
    x_train = value
    y_train = np.full(len(x_train), key)
    # train_y = arr[:, 0]   # Select the first column (index 0)
    n_train, p = x_train.shape

    c = 100
    epsilon = 1e-5
    maxIter = 1e3
    maxInnerIter = 5

    # Loop through a set of variables
    Ks = [10]  # Add more values if needed
    lambdas = [1]  # Add more values if needed
    rs = [0.3]  # Add more values if needed

    for K in Ks:
        for lambda_ in lambdas:
            for r in rs:
                q = int(np.ceil(p * r))
                runid = f'{dataset_name}_l_{lambda_}_K_{K}_q_{q}'

                # Train SIDL on training set
                start_time = time.time()
                S, A, Offsets, F_obj = utils.USIDL_with_alpha_const(x_train, y_train, lambda_, K, q, c, epsilon, maxIter, maxInnerIter, runid)
                learn_time = time.time() - start_time
                print(f'\n##### TRAINING TIME on TRAIN SET (K={K}, lambda={lambda_}, r={r}): {learn_time} secs.\n\n')
                all_shapelets.append(S)
                shapelet_dict[key] = S

all_shapelets_array = np.vstack(all_shapelets)
all_shapelets_array.shape
base_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/deep_analysis/UCR_other_datasets"
save_dict_path = os.path.join(base_path, f"{dataset_name}.pkl")
print(save_dict_path)
with open(save_dict_path, 'wb') as f:
    pickle.dump(shapelet_dict, f)


shapelet_features_train = utils.shapelet_transform(train_X, all_shapelets_array)
shapelet_features_test = utils.shapelet_transform(test_X, all_shapelets_array)

print(shapelet_features_train.shape)
print(shapelet_features_test.shape)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(shapelet_features_train, train_y)

# Evaluate model
y_pred = clf.predict(shapelet_features_test)
print(f"Classification Accuracy using RF classifier using shapelet transformed features: {accuracy_score(test_y, y_pred) * 100:.2f}%")

svm_model = SVC(kernel="linear", C=1.0, gamma="scale", random_state=42)
svm_model.fit(shapelet_features_train, train_y)
y_pred = svm_model.predict(shapelet_features_test)
print(f"Classification Accuracy using SVM using shapelet transformed features: {accuracy_score(test_y, y_pred) * 100:.2f}%")

svm_model = SVC(kernel="linear", C=1.0, gamma="scale", random_state=42)
svm_model.fit(train_X, train_y)
y_pred = svm_model.predict(test_X)
print(f"Classification Accuracy using SVM using raw features: {accuracy_score(test_y, y_pred) * 100:.2f}%")


