import numpy as np
from time import time
from typing import *


class KNN:
    """ K Nearest Neighbors Algorithm. """

    accuracy: float
    classwise_accuracy: List[float]
    prediction_time: float
    predicted_y: np.ndarray
    k: int
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    predicted_y: np.ndarray
    unique_classes: np.ndarray

    def __init__(self, train: np.ndarray, k: int) -> None:
        # Split train dataset into x and y
        self.train_x, self.train_y = self.x_y_split(train)

        self.k = k
        self.unique_classes = np.unique(self.train_y)

    @staticmethod
    def x_y_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        return dataset[:, :-1], dataset[:, -1].astype(int)

    @staticmethod
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def find_neighbors(self, test_x: np.ndarray) -> np.array:
        # num_test x num_train: distances from each test point to every train point
        distances = [self.euclidean_distance(test_point, self.train_x) for test_point in test_x]
        # Get k nearest neighbors for each point
        neighbors = []
        # neighbor_distances = []
        for point_distance in distances:
            # Sort distances and get indices
            curr_sorted_distance_idx = np.argsort(point_distance)
            # Get k shortest distance indices
            curr_closest_neighbors = curr_sorted_distance_idx[:self.k + 1]
            # Get k shortest distances
            # curr_closest_neighbor_distances = point_distance[curr_closest_neighbors]
            # Save them to lists
            neighbors.append(curr_closest_neighbors)
            # neighbor_distances.append(curr_closest_neighbor_distances)

        return np.array(neighbors)

    def fit(self, test: np.ndarray):
        self.test_x, self.test_y = self.x_y_split(test)
        start = time()
        # Get neighbors from the train set
        neighbors = self.find_neighbors(self.test_x)
        # Predict the classes
        predicted_y = [np.argmax(np.bincount(self.train_y[neighbor])) for neighbor in neighbors]
        self.predicted_y = np.array(predicted_y)
        # Save time
        self.prediction_time = time() - start

    def get_statistics(self) -> Tuple[float, List[float], float]:

        self.accuracy = np.count_nonzero(self.predicted_y == self.test_y) / len(self.predicted_y)
        self.classwise_accuracy = []
        for class_n in self.unique_classes:
            test_y_current = self.test_y[self.test_y == self.unique_classes[class_n]]
            predicted_y_current = self.predicted_y[self.test_y == self.unique_classes[class_n]]
            current_acc = np.count_nonzero(predicted_y_current == test_y_current) / len(
                predicted_y_current)
            self.classwise_accuracy.append(current_acc)

        return self.accuracy, self.classwise_accuracy, self.prediction_time
