import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict


class ShapeletQuality_Fstat:

    # The F-statistic measures the ratio of between-group variability to
    # within-group variability. Higher values indicate better discrimination
    # between classes.


    def __init__(self):
        self.distances = None
        self.class_labels = None
        self.classes = None
        self.n_classes = None
        self.n_series = None

    def calculate_f_statistic(self, distances: List[float], class_labels: List) -> float:
        # List of distances from candidate shapelet to each time series
        # list of corresponding class labels for each time series

        # Store data
        self.distances = np.array(distances)
        self.class_labels = np.array(class_labels)
        self.classes = np.unique(class_labels)
        self.n_classes = len(self.classes)  # C in the formula
        self.n_series = len(distances)  # n in the formula

        # Split distances by class membership
        distances_by_class = self._split_distances_by_class()

        #  class means (D̄ᵢ)
        class_means = self._calculate_class_means(distances_by_class)

        #  overall mean (D̄)
        overall_mean = np.mean(self.distances)

        #  numerator
        between_ss = self._calculate_between_group_ss(class_means, overall_mean, distances_by_class)

        # denominator
        within_ss = self._calculate_within_group_ss(distances_by_class, class_means)

        # Calculate F-statistic
        f_statistic = between_ss / within_ss

        return f_statistic

    def _split_distances_by_class(self) -> Dict:
        """Split distances by class membership."""
        distances_by_class = defaultdict(list)

        for distance, label in zip(self.distances, self.class_labels):
            distances_by_class[label].append(distance)

        return distances_by_class

    def _calculate_class_means(self, distances_by_class: Dict) -> Dict:
        class_means = {}
        for class_label, distances in distances_by_class.items():
            class_means[class_label] = np.mean(distances)
        return class_means

    def _calculate_between_group_ss(self, class_means: Dict, overall_mean: float,
                                    distances_by_class: Dict) -> float:

        # numerator calculation : Σᵢ(D̄ᵢ - D̄)² / (C - 1)

        between_ss = 0.0

        for class_label in self.classes:
            class_mean = class_means[class_label]
            term = (class_mean - overall_mean) ** 2
            between_ss = between_ss + term

            # Divide by degrees of freedom (C - 1)
        between_ss = between_ss / (self.n_classes - 1)

        return between_ss

    def _calculate_within_group_ss(self, distances_by_class: Dict,
                                   class_means: Dict) -> float:

        # denominator calculation: ΣᵢΣⱼ∈Dᵢ(dⱼ - D̄ᵢ)² / (n - C)
        within_ss = 0.0

        for class_label in self.classes:
            class_distances = distances_by_class[class_label]
            class_mean = class_means[class_label]

            for distance in class_distances:
                within_ss += (distance - class_mean) ** 2

        # Divide by degrees of freedom (n - C)
        within_ss = within_ss / (self.n_series - self.n_classes)

        return within_ss

    def assess_multiple_shapelets(self, shapelet_distances: Dict,
                                  class_labels: List) -> pd.DataFrame:

        results = []

        for shapelet_id, distances in shapelet_distances.items():
            f_stat = self.calculate_f_statistic(distances, class_labels)

            results.append({
                'shapelet_id': shapelet_id,
                'f_statistic': f_stat,
                'quality_rank': None  # Will be filled after sorting
            })

        # Convert to df and rank by F-stat value
        df = pd.DataFrame(results)
        df = df.sort_values('f_statistic', ascending=False)
        df['quality_rank'] = range(1, len(df) + 1)
        df = df.reset_index(drop=True)

        return df


