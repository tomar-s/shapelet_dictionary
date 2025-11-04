import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
from metrics import dtw, ed
from sklearn.model_selection import train_test_split

def load_csv(filename, delimiter=',', skip_header=0, dtype = object):
    try:
        data = np.genfromtxt(filename, delimiter=delimiter, skip_header=skip_header, dtype=dtype)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def split_train_test_and_normalise(X, Y):

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for label in np.unique(Y):
        x = X[np.where(Y == label)[0]]
        y = Y[np.where(Y == label)[0]]
        temp_train, temp_train_y= x[:len(x)//2], y[:len(x)//2]
        temp_test, temp_test_y = x[len(x) // 2:], y[len(x) // 2:]

        # perform z normalisation of data splits before learning shapelets
        mean_train = np.mean(temp_train)
        temp_train_scaled = (temp_train - mean_train) / np.std(temp_train)
        # mean_test = np.mean(temp_test)
        temp_test_scaled = (temp_test - mean_train) / np.std(temp_train)

        train_data.extend(temp_train_scaled)
        train_labels.extend(temp_train_y)
        test_data.extend(temp_test_scaled)
        test_labels.extend(temp_test_y)

    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

def split_train_test(X, Y):

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for label in np.unique(Y):
        x = X[np.where(Y == label)[0]]
        y = Y[np.where(Y == label)[0]]
        temp_train, temp_train_y= x[:len(x)//2], y[:len(x)//2]
        temp_test, temp_test_y = x[len(x) // 2:], y[len(x) // 2:]

        train_data.extend(temp_train)
        train_labels.extend(temp_train_y)
        test_data.extend(temp_test)
        test_labels.extend(temp_test_y)

    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

def split_train_test_offline(data_df, test_size, random_state):

    # when data is already in TS window format, split into train and test with equal percentage of classes

    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    X = data_df.drop('label', axis=1)
    y = data_df["label"].values
    unique_classes = np.unique(y)

    for label in unique_classes:
        X_cls = data_df[data_df['label'] == label].drop(columns='label').values
        y_cls = data_df[data_df['label'] == label]["label"].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_cls, y_cls,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=None  # Not needed as we're splitting within class
        )

        X_train_list.append(X_tr)
        X_test_list.append(X_te)
        y_train_list.append(y_tr)
        y_test_list.append(y_te)

    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # #  shuffling the final set
    # train_perm = np.random.permutation(len(X_train))
    # test_perm = np.random.permutation(len(X_test))

    return X_train, y_train, X_test, y_test


# Bootstrapped subsequence sampling from a stepped sliding window, with a random initialization
def bootstrapped(X, Y, window=100, step=1, qty=1000):
    data = []
    labels = []
    for label in np.unique(Y):
        x = X[np.where(Y == label)[0]]
        y = Y[np.where(Y == label)[0]]
        temp_data = []
        temp_labels = []
        if len(x) > 0:
            a = 0
            while len(temp_data) < qty:
                rano = np.random.randint(0, len(x)) if a != 0 else a
                a += 1
                for i in range(rano, len(x) - window, step):
                    if len(temp_data) >= qty:
                        break
                    temp_labels.append(y[i])
                    temp_data.append(np.array(x[i:i + window]))
            data.extend(temp_data[:qty])
            labels.extend(temp_labels[:qty])

    return np.array(data), np.array(labels)


# Simple stepped sliding window for subsequence extraction
def slide(X, Y, window=100, step=1, qty=1000):
    data = []
    labels = []
    for label in np.unique(Y):
        x = X[np.where(Y == label)[0]]
        y = Y[np.where(Y == label)[0]]
        temp_data = []
        temp_labels = []
        for i in range(0, len(x) - window, step):
            if len(temp_data) >= qty:
                break
            temp_data.append(np.array(x[i:i + window]))
            temp_labels.append(y[i])
        data.extend(temp_data[:qty])
        labels.extend(temp_labels[:qty])

    return np.array(data), np.array(labels)


# Extracts randomly positioned subsequences
def random(X, Y, window=100, qty=1000):
    data = []
    labels = []
    for label in np.unique(Y):
        x = X[np.where(Y == label)[0]]
        y = Y[np.where(Y == label)[0]]
        if len(x) > window:
            for n in range(qty):
                indexer = np.random.randint(window // 2, len(x) - window // 2)
                series = x[indexer - window // 2:indexer + window // 2]
                data.append(np.array(series))
                labels.append(y[indexer])

    return np.array(data), np.array(labels)


def data_compiler(X, y, method='slide', window=100, step=1, qty=1000):
    if method == 'slide':
        return slide(X, y, window=window, step=step, qty=qty)

    elif method == 'random':
        return random(X, y, window=window, qty=qty)

    elif method == 'bootstrapped':
        return bootstrapped(X, y, window=window, step=step, qty=qty)

    else:
        raise ValueError('Only slide, random and bootstrapped methods are supported')


def plot_shapelets_per_label(shapelets_dict, max_shapelets=5):
    """
    Plots learned shapelets per class label.

    Parameters:
        shapelets_dict (dict): Dictionary with labels as keys and shapelets (arrays) as values.
        max_shapelets (int): Max number of shapelets to plot per label.
    """
    num_labels = len(shapelets_dict)
    fig, axes = plt.subplots(num_labels, max_shapelets, figsize=(max_shapelets * 3, num_labels * 2))

    if num_labels == 1:
        axes = [axes]  # In case only one label

    for i, (label, shapelet_list) in enumerate(shapelets_dict.items()):
        for j in range(min(len(shapelet_list), max_shapelets)):
            ax = axes[i][j] if num_labels > 1 else axes[0][j]
            ax.plot(shapelet_list[j])
            ax.set_title(f"Label {label} - Shapelet {j+1}")
            ax.set_xlim(0, len(shapelet_list[j]))  # Adjust x-axis as needed
        # Hide unused subplots
        for j in range(len(shapelet_list), max_shapelets):
            ax = axes[i][j] if num_labels > 1 else axes[0][j]
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def data_shapelet_dist_transform(data, shapelets):
    """
       calculates the dtw distance between shapelets and each time series.

       Parameters:
           shapelets_dict (dict): Dictionary with labels as keys and shapelets (arrays) as values.
          data (nd array): train data as numpy array
       """

    all_shapelets = [s for arr in shapelets.values() for s in arr]
    transformed_data = np.zeros((data.shape[0], len(all_shapelets)))

    for i, time_series in enumerate(data):
        for j, shapelet in enumerate(all_shapelets):
            transformed_data[i, j] = dtw(time_series, shapelet)

    return transformed_data
