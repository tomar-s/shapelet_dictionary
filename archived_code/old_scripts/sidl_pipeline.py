from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from shapelet_transform import ShapeletTransform
from archived_code import utils


def build_pipeline(X_train):

    shapelet_dict = utils.learn_shapelet_dict_via_sidl(X_train)
    all_shapelets = [s for shapelet_list in shapelet_dict.values() for s in shapelet_list]

    pipeline = Pipeline([
        ("shapelet_transform", ShapeletTransform(all_shapelets)),
        ("classifier", MLPClassifier(random_state=42))
    ])

    return pipeline