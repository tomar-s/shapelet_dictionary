import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score
from helper.preprocess_data import split_train_test_offline
from helper.utils import learn_shapelet_dict_via_sidl
from helper.shapelet_transform import ShapeletTransform
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def create_classifier(classifier_name):
    if classifier_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=35)
    if classifier_name == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant',  learning_rate_init=0.001, max_iter=1000, random_state=35, verbose=True)
    if classifier_name == 'sgd':
        return SGDClassifier(loss= "log_loss", random_state=35)
    if classifier_name == 'svm':
        return SVC(kernel="linear", C=1.0, gamma="scale", max_iter=1000,  random_state=35)


def main():

    classifiers = ['random_forest', 'mlp', 'sgd', 'svm']
    np.random.seed(33)
    data_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/datasets/HAR/balanced_data_stream_8_batches.csv"
    har_data = pd.read_csv(data_path)
    # labels = har_data["label"]
    # encoder = LabelEncoder()
    # encoder.fit(labels)
    # labels_encoded = encoder.transform(labels)

    train_x, train_y, test_x, test_y = split_train_test_offline(har_data, 0.5, random_state=35)

    mean_train = train_x.mean(axis=1, keepdims=True)
    std_train = train_x.std(axis=1, keepdims=True)
    std_train[std_train == 0] = 1.0
    X_train_norm = (train_x - mean_train) / std_train
    X_test_norm = (test_x - mean_train) / std_train

    #  only for shapelet learning combine X_train_norm with train_y to learn shapelets class wise
    train_data_for_shapelet = np.hstack((X_train_norm, train_y.reshape(-1, 1)))

    shapelets_dict = learn_shapelet_dict_via_sidl(train_data_for_shapelet)
    all_shapelets = [s for s_list in shapelets_dict.values() for s in s_list]

    # base_path = "/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/deep_analysis/HAR_offline/HAR_dict.pkl"
    # # save_dict_path = os.path.join(base_path, f"{dataset_name}.pkl")
    # # print(save_dict_path)
    # with open(base_path, 'wb') as f:
    #     pickle.dump(shapelets_dict, f)

    transformer = ShapeletTransform(all_shapelets)
    X_transformed_train = transformer.transform(X_train_norm)
    print(X_transformed_train.shape)
    X_transformed_test = transformer.transform(X_test_norm)
    print(X_transformed_test.shape)

    for classifier in classifiers:

        clf = create_classifier(classifier)
        # clf = SVC(kernel="linear", C=1.0, gamma="scale", max_iter=1000,  random_state=42)
        clf.fit(X_transformed_train, train_y)
        y_pred = clf.predict(X_transformed_test)
        print(f"Classification Accuracy using {classifier} with shapelet features: {accuracy_score(test_y, y_pred) * 100:.2f}%")
        print(classification_report(test_y, y_pred))
        f1 = f1_score(test_y, y_pred, average='macro', zero_division=0)
        print("f1 score", f1)
        precision = precision_score(test_y, y_pred, average='macro', zero_division=0)
        print("precision", precision)

    # with open('/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/deep_analysis/HAR_offline/HAR_prediction_results.pkl', 'wb') as f:
    #     pickle.dump({'y_pred': y_pred, 'y_test': test_y, 'X_test': X_test_norm}, f)

    print("done")

if __name__ == "__main__":
    main()