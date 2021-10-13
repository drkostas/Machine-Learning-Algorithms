import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from time import time
from typing import *
from custom_libs import ColorizedLogger, timeit

logger = ColorizedLogger('Project2 Models', 'green')


class KNN:
    """ K Nearest Neighbors Algorithm. """

    accuracy: float
    classwise_accuracy: List[float]
    prediction_time: float
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
    def euclidean_distance(a: np.ndarray, b: np.ndarray):
        dist = np.sqrt(np.sum((a - b) ** 2, axis=1))
        return dist

    def find_neighbors(self) -> np.array:
        # num_test x num_train: distances from each test point to every train point
        distances = [self.euclidean_distance(test_point, self.train_x) for test_point in self.test_x]
        # Get k nearest neighbors for each point
        neighbors = []
        for point_distance in distances:
            # Sort distances and get indices
            curr_sorted_distance_idx = np.argsort(point_distance)
            # Get k shortest distance neighbors
            curr_closest_neighbors = curr_sorted_distance_idx[:self.k]
            neighbors.append(curr_closest_neighbors)
        return np.array(neighbors)

    def predict(self, test: np.ndarray, only_x: bool = False):
        if only_x:
            self.test_x = test
        else:
            self.test_x, self.test_y = self.x_y_split(test)
        start = time()
        # Get neighbors from the train set
        neighbors = self.find_neighbors()
        # Predict the classes
        predicted_y = [np.argmax(np.bincount(self.train_y[neighbor])) for neighbor in neighbors]
        self.predicted_y = np.array(predicted_y)
        # Save time
        self.prediction_time = time() - start

        return self.predicted_y

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

    def print_statistics(self, name: str) -> None:
        # Check if statistics have be calculated
        if any(v is None for v in [self.accuracy, self.classwise_accuracy, self.prediction_time]):
            self.get_statistics()
        logger.info(f"kNN (k={self.k}) for the {name} dataset")
        logger.info(f"The overall accuracy is: {self.accuracy:.4f}")
        logger.info(f"The classwise accuracies are: {self.classwise_accuracy}")
        logger.info(f"Total time: {self.prediction_time:.4f} sec(s)")


class Kmeans:
    """ K Means Algorithm. """

    accuracy: float
    classwise_accuracy: List[float]
    prediction_time: float
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    predicted_y: np.ndarray
    centroids: np.ndarray
    cluster_assignments: np.ndarray
    unique_classes: np.ndarray
    num_classes: int
    total_epochs: int
    membership_changes: List[int]

    def __init__(self, train_data: np.ndarray, seed: int = None) -> None:
        # Split train dataset into x and y
        self.train_x, self.train_y = self.x_y_split(train_data)
        self.unique_classes = np.unique(self.train_y)
        self.num_classes = len(self.unique_classes)
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def x_y_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        return dataset[:, :-1], dataset[:, -1].astype(int)

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray):
        n = a.shape[0]
        m = b.shape[0]
        a_dot_sum = (a * a).sum(axis=1).reshape((n, 1)) * np.ones(shape=(1, m))
        b_dot_sum = (b * b).sum(axis=1) * np.ones(shape=(n, 1))
        d_sq = a_dot_sum + b_dot_sum - 2 * a.dot(b.T)

        return d_sq

    def fit(self):
        """ Initialize centroids randomly as distinct elements of features. """
        num_train_points = self.train_x.shape[0]  # num sample points
        centroid_ids = np.random.choice(num_train_points, (self.num_classes,), replace=False)
        self.centroids = self.train_x[centroid_ids, :]

    def predict(self, test_data: np.ndarray, dist: str = 'custom'):
        # Get appropriate distance metric
        if dist == 'custom':
            dist_calc = self.euclidean_distance
        elif dist == 'cdist_sq':
            dist_calc = lambda features, centroids: cdist(features, centroids, 'sqeuclidean')
        elif dist == 'cdist':
            dist_calc = lambda features, centroids: cdist(features, centroids, 'euclidean')
        else:
            raise Exception('Error: dist should be either custom, cdist, or cdist_sq')
        # Split test set in x and y
        self.test_x, self.test_y = self.x_y_split(test_data)
        # Run Kmeans and time it
        timeit_fit = timeit(internal_only=True)
        with timeit_fit:
            self._run(dist_calc=dist_calc)
        self.prediction_time = timeit_fit.total

        # Fix class mismatch (for 2 classes only)
        self.accuracy = np.count_nonzero(self.predicted_y == self.test_y) / len(self.predicted_y)
        if self.accuracy < 0.5 and self.num_classes == 2:
            self.predicted_y = 1 - self.predicted_y
            self.accuracy = np.count_nonzero(self.predicted_y == self.test_y) / len(self.predicted_y)

        return self.predicted_y

    def _run(self, dist_calc: Callable) -> None:
        """ Run k-means algorithm to convergence. """

        # Loop until convergence
        self.total_epochs = 1
        self.predicted_y = np.zeros(self.test_x.shape[0], dtype=np.uint8)
        self.membership_changes = []
        while True:
            self.total_epochs += 1
            # Compute distances from sample points to centroids
            # all pair-wise _squared_ distances
            centroid_distances = dist_calc(self.test_x, self.centroids)

            # Expectation step: assign clusters
            previous_assignments = self.predicted_y
            self.predicted_y = np.argmin(centroid_distances, axis=1)

            # Maximization step: Update centroid for each cluster
            for cluster_ind in range(self.num_classes):
                features_of_curr_cluster = self.test_x[self.predicted_y == cluster_ind]
                self.centroids[cluster_ind, :] = np.mean(features_of_curr_cluster, axis=0)
            # Break Condition
            self.membership_changes.append(np.count_nonzero(self.predicted_y != previous_assignments))
            if (self.predicted_y == previous_assignments).all():
                break

    def get_statistics(self) -> Tuple[float, List[float], float, int, List[int]]:

        self.classwise_accuracy = []
        for class_n in self.unique_classes:
            test_y_current = self.test_y[self.test_y == self.unique_classes[class_n]]
            predicted_y_current = self.predicted_y[self.test_y == self.unique_classes[class_n]]
            current_acc = np.count_nonzero(predicted_y_current == test_y_current) / len(
                predicted_y_current)
            self.classwise_accuracy.append(current_acc)
        statistics = (self.accuracy, self.classwise_accuracy, self.prediction_time,
                      self.total_epochs, self.membership_changes)
        return statistics

    def print_statistics(self, name: str) -> None:
        # Check if statistics have be calculated
        if any(v is None for v in [self.accuracy, self.classwise_accuracy]):
            self.get_statistics()
        # Print Statistics
        logger.info(f"Kmeans for the {name} dataset")
        logger.info(f"The overall accuracy is: {self.accuracy:.4f}")
        logger.info(f"The classwise accuracies are: {self.classwise_accuracy}")
        logger.info(f"Total Iterations: {self.total_epochs}")
        logger.info(f"Total time: {self.prediction_time:.4f} sec(s)")


class WTA:
    """ Winner Takes All Algorithm. """

    accuracy: float
    classwise_accuracy: List[float]
    prediction_time: float
    x: np.ndarray
    y: np.ndarray
    predicted_y: np.ndarray
    centroids: np.ndarray
    unique_classes: np.ndarray
    num_classes: int
    total_epochs: int
    epsilon: float
    membership_changes: List[int]

    def __init__(self, train_data: np.ndarray, seed: int = None) -> None:
        # Split train dataset into x and y
        self.train_x, self.train_y = self.x_y_split(train_data)
        self.unique_classes = np.unique(self.train_y)
        self.num_classes = len(self.unique_classes)
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def x_y_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        return dataset[:, :-1], dataset[:, -1].astype(int)

    def fit(self):
        # initialize centroids randomly as distinct elements of features
        num_train_points = self.train_x.shape[0]  # num sample points
        centroid_ids = np.random.choice(num_train_points, (self.num_classes,), replace=False)
        self.centroids = self.train_x[centroid_ids, :]

    def predict(self, test_data: np.ndarray, epsilon: float = 0.01):
        # Split test set
        self.test_x, self.test_y = self.x_y_split(test_data)
        # Run WTA and time it
        timeit_fit = timeit(internal_only=True)
        with timeit_fit:
            self._run(epsilon=epsilon)
        self.prediction_time = timeit_fit.total
        # Fix class mismatch (for 2 classes only)
        self.accuracy = np.count_nonzero(self.predicted_y == self.test_y) / len(self.predicted_y)
        if self.accuracy < 0.5 and len(self.unique_classes) == 2:
            self.predicted_y = 1 - self.predicted_y
            self.accuracy = np.count_nonzero(self.predicted_y == self.test_y) / len(self.predicted_y)

        self.epsilon = epsilon
        return self.predicted_y

    def _run(self, epsilon: float = 0.01):
        num_test_points = self.test_x.shape[0]
        self.predicted_y = np.zeros(num_test_points, dtype=np.uint8)
        self.total_epochs = 1
        self.membership_changes = []
        # Loop until convergence
        while True:
            self.total_epochs += 1
            dist = np.empty((num_test_points, 0))
            for j in range(self.centroids.shape[0]):
                d = self.test_x - self.centroids[j, :]
                d = np.linalg.norm(d, axis=1)
                dist = np.column_stack((dist, d))
            previous_assignments = self.predicted_y
            self.predicted_y = np.argmin(dist, axis=1)
            df = pd.DataFrame(np.column_stack((self.test_x, self.predicted_y)))
            cen = df.groupby([df.iloc[:, -1]], as_index=False).mean()
            # Modify Centroids
            for j in range(cen.shape[0]):
                self.centroids[int(cen.iloc[int(j), -1]), :] \
                    += epsilon*(cen.iloc[j, :-1].values-self.centroids[int(cen.iloc[int(j), -1]), :])

            # Break Condition
            self.membership_changes.append(np.count_nonzero(self.predicted_y != previous_assignments))
            if (self.predicted_y == previous_assignments).all():
                break

    def get_statistics(self) -> Tuple[float, List[float], float, int, List[int]]:

        self.classwise_accuracy = []
        for class_n in self.unique_classes:
            test_y_current = self.test_y[self.test_y == self.unique_classes[class_n]]
            predicted_y_current = self.predicted_y[self.test_y == self.unique_classes[class_n]]
            current_acc = np.count_nonzero(predicted_y_current == test_y_current) / len(
                predicted_y_current)
            self.classwise_accuracy.append(current_acc)
        statistics = (self.accuracy, self.classwise_accuracy, self.prediction_time,
                      self.total_epochs, self.membership_changes)

        return statistics

    def print_statistics(self, name: str) -> None:
        # Check if statistics have be calculated
        if any(v is None for v in [self.accuracy, self.classwise_accuracy]):
            self.get_statistics()
        logger.info(f"WTA for the {name} dataset")
        logger.info(f"The overall accuracy is: {self.accuracy:.4f}")
        logger.info(f"The classwise accuracies are: {self.classwise_accuracy}")
        logger.info(f"Total Iterations: {self.total_epochs}")
        logger.info(f"Total time: {self.prediction_time:.4f} sec(s)")
