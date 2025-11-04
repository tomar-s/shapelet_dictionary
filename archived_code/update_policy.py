import numpy as np
import pandas as pd
import time
import heapq
from scipy.io import loadmat, savemat, arff
from scipy.spatial.distance import cdist


def euclidean_distance(sequence_s, sequence_r):
    """
    Calculate the Euclidean distance between two subsequences of equal length.

    Parameters:
    sequence_s (array-like): First subsequence
    sequence_r (array-like): Second subsequence

    Returns:
    float: The Euclidean distance between the subsequences
    """
    # Convert inputs to numpy arrays for efficient calculation
    s = np.array(sequence_s)
    r = np.array(sequence_r)

    # Check if sequences have the same length
    if len(s) != len(r):
        raise ValueError("Sequences must have the same length")

    # Calculate Euclidean distance
    distance = np.sqrt(np.sum((s - r) ** 2))

    return distance

def find_closest_shapelets(shapelets_a, shapelets_b, top_k=1):
    """
    For each shapelet in set A, find the top_k closest shapelets from set B.

    Parameters:
    shapelets_a (list): List of reference shapelets
    shapelets_b (list): List of candidate shapelets
    top_k (int): Number of closest shapelets to find for each shapelet in A

    Returns:
    dict: Dictionary mapping each shapelet index in A to a list of (distance, shapelet_index) tuples
    """
    closest_matches = {}

    for i, shapelet_a in enumerate(shapelets_a):
        # Calculate distances to all shapelets in B
        distances = []
        for j, shapelet_b in enumerate(shapelets_b):
            try:
                dist = euclidean_distance(shapelet_a, shapelet_b)
                distances.append((dist, j))
            except ValueError as e:
                print(f"Error comparing shapelet A[{i}] with B[{j}]: {e}")
                continue

        # Get the top_k closest shapelets
        top_matches = heapq.nsmallest(top_k, distances, key=lambda x: x[0])
        closest_matches[i] = top_matches

    return closest_matches

def combine_shapelet_sets(shapelets_a, shapelets_b, closest_matches, top_per_shapelet=1):
    """
    Combine shapelet set A with the closest matches from set B.

    Parameters:
    shapelets_a (list): List of reference shapelets
    shapelets_b (list): List of candidate shapelets
    closest_matches (dict): Dictionary mapping each shapelet index in A to a list of (distance, shapelet_index) tuples
    top_per_shapelet (int): Number of closest matches to include for each shapelet in A

    Returns:
    list: Combined list of shapelets
    """
    # Start with all shapelets from set A
    combined_shapelets = shapelets_a.copy()

    # Track indices of shapelets from B that we've already added
    added_indices = set()

    # For each shapelet in A, add its closest matches from B
    for i in range(len(shapelets_a)):
        if i in closest_matches:
            for j in range(min(top_per_shapelet, len(closest_matches[i]))):
                _, b_idx = closest_matches[i][j]
                if b_idx not in added_indices:
                    combined_shapelets.append(shapelets_b[b_idx])
                    added_indices.add(b_idx)

    return combined_shapelets


def combine_top_10_closest_shapelets(shapelets_a, shapelets_b):
    """
    Find the 10 closest shapelets from B for set A and combine them.

    Parameters:
    shapelets_a (numpy.ndarray or list): Array of reference shapelets (size 10)
    shapelets_b (numpy.ndarray or list): Array of candidate shapelets (size 20)

    Returns:
    numpy.ndarray: Combined array of shapelets (size 20)
    """
    # Convert to numpy arrays if they aren't already
    shapelets_a = np.array(shapelets_a)
    shapelets_b = np.array(shapelets_b)

    # Calculate all pairwise distances
    all_distances = []

    for i, shapelet_a in enumerate(shapelets_a):
        for j, shapelet_b in enumerate(shapelets_b):
            try:
                dist = euclidean_distance(shapelet_a, shapelet_b)
                all_distances.append((dist, i, j))
            except ValueError as e:
                print(f"Error comparing shapelet A[{i}] with B[{j}]: {e}")
                continue

    # Sort by distance
    all_distances.sort(key=lambda x: x[0])

    # Get indices of top 10 closest shapelets from B
    selected_indices = set()
    top_matches = []

    for dist, i, j in all_distances:
        if j not in selected_indices:
            selected_indices.add(j)
            top_matches.append((dist, j))
            if len(selected_indices) >= 12:
                break

    # Extract the indices of the selected shapelets from B
    selected_b_indices = [j for _, j in top_matches]

    # Combine shapelets using concatenate function
    combined_shapelets = np.concatenate([shapelets_a, shapelets_b[selected_b_indices]], axis=0)

    return combined_shapelets

def combine_top_10_dissimilar_shapelets(shapelets_a, shapelets_b):
    """
    Find the 10 most dissimilar shapelets from B compared to set A and combine them.

    Parameters:
    shapelets_a (numpy.ndarray or list): Array of reference shapelets (size 10) which are from new distribution
    shapelets_b (numpy.ndarray or list): Array of candidate shapelets (size 20) which belong to older distribution

    Returns:
    numpy.ndarray: Combined array of shapelets (size 20)
    """
    # Convert to numpy arrays if they aren't already
    shapelets_a = np.array(shapelets_a)
    shapelets_b = np.array(shapelets_b)

    # For each shapelet in B, find its minimum distance to any shapelet in A
    min_distances = []

    for j, shapelet_b in enumerate(shapelets_b):
        # Calculate distances to all shapelets in A
        distances = []
        for i, shapelet_a in enumerate(shapelets_a):
            try:
                dist = euclidean_distance(shapelet_a, shapelet_b)
                distances.append(dist)
            except ValueError as e:
                print(f"Error comparing shapelet A[{i}] with B[{j}]: {e}")
                continue

        # Find the minimum distance to any shapelet in A
        if distances:
            min_dist = min(distances)
            min_distances.append((min_dist, j))
        else:
            print(f"Warning: No valid distances for shapelet B[{j}]")

    # Sort by distance in descending order to get the most dissimilar first
    min_distances.sort(key=lambda x: x[0], reverse=True)

    # Get indices of the 10 most dissimilar shapelets from B
    selected_b_indices = [j for _, j in min_distances[:12]]

    # Combine shapelets using numpy's concatenate function
    combined_shapelets = np.concatenate([shapelets_a, shapelets_b[selected_b_indices]], axis=0)

    return combined_shapelets

def combine_random_shapelets(shapelets_a, shapelets_b):
    """
        Find 10 random shapelets from B and combine them to set A.

        Parameters:
        shapelets_a (numpy.ndarray or list): Array of reference shapelets (size 10) which are from new distribution
        shapelets_b (numpy.ndarray or list): Array of candidate shapelets (size 20) which belong to older distribution

        Returns:
        numpy.ndarray: Combined array of shapelets (size 20)
        """
    # Convert to numpy arrays if they aren't already

    shapelets_a = np.array(shapelets_a)
    shapelets_b = np.array(shapelets_b)

    S_random = shapelets_b[np.random.choice(shapelets_b.shape[0], 10, replace=False)]
    combined_shapelets = np.vstack((S_random, shapelets_a))

    return combined_shapelets

def combine_topK_previous(shapelet_a, shapelet_b, overall_ranks):
    """
        Find top k shapelets from B which are ranked based on sparse coefficient matrix and combine them to set A.

        Parameters:
        shapelets_a (numpy.ndarray or list): Array of reference shapelets (size 10) which are from new distribution
        shapelets_b (numpy.ndarray or list): Array of candidate shapelets (size 20) which belong to older distribution

        Returns:
        numpy.ndarray: Combined array of shapelets (size 20)
        """
    top10_indices = overall_ranks[:10]
    S_top10 = shapelet_b[top10_indices, :]

    S_updated_topk = np.vstack((S_top10, shapelet_a))

    return S_updated_topk


def combine_similar_labelwise (shapelet_dict_old, shapelet_dict_new):
    "labelwise comparison of dictionary shapelets to update the dictionary"

    updated_shapelets_dict = {}
    for label in shapelet_dict_old.keys():
        old = np.array(shapelet_dict_old[label])
        new = np.array(shapelet_dict_new[label])

        all_distances = []
        for i, shapelet_old in enumerate(old):
            # calculating pairwise distances
            for j, shapelet_new in enumerate(new):
                dist = euclidean_distance(shapelet_old, shapelet_new)
                all_distances.append((dist, i, j))

        all_distances.sort(key=lambda x: x[0])
        # selecting top N pairs with unique old and new indices
        top_matches = []
        combined_shapelets = []
        used_a = set()
        used_b = set()
        N = len(old)

        for dist, i, j in all_distances:
            if i not in used_a and j not in used_b:
                top_matches.append((dist, i, j))
                combined_shapelets.append(old[i])
                combined_shapelets.append(new[j])
                used_a.add(i)
                used_b.add(j)
            if len(combined_shapelets) >= N:
                break
        updated_shapelets_dict[label] = combined_shapelets

    return updated_shapelets_dict




