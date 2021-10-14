import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from time import time
from typing import *
from custom_libs import ColorizedLogger, timeit

logger = ColorizedLogger('Project3 Models', 'green')


class FLD:
    """ Fischer's Linear Discriminant. """
    w: np.ndarray

    def __init__(self) -> None:
        pass

    def fit(self, data: np.ndarray):
        data_x_c1, data_x_c2 = self.two_classes_split(dataset=data)
        # Calculate class means
        means_c1 = np.expand_dims(np.mean(data_x_c1, axis=0), axis=1)
        means_c2 = np.expand_dims(np.mean(data_x_c2, axis=0), axis=1)
        # Calculate s1, s2
        data_minus_means_c1 = (means_c1.T - data_x_c1).T
        s1 = data_minus_means_c1 @ data_minus_means_c1.T
        data_minus_means_c2 = (means_c2.T - data_x_c2).T
        s2 = data_minus_means_c2 @ data_minus_means_c2.T
        # Calculate Sw, Sw inverse
        sw = s1 + s2
        sw_inv = np.linalg.inv(sw)
        # Calculate the difference of the two means (equivalent to Sb)
        class_means_diff = means_c1 - means_c2
        # Calculate the projection vector
        self.w = sw_inv @ class_means_diff

    def transform(self, data: np.ndarray) -> np.array:
        data_x_c1, data_x_c2 = self.two_classes_split(dataset=data)
        data_x_proj_c1 = data_x_c1 @ self.w
        data_x_proj_c2 = data_x_c2 @ self.w
        return self.two_classes_merge(data_x_proj_c1, data_x_proj_c2, data[:, -1][:, np.newaxis])

    @staticmethod
    def x_y_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        return dataset[:, :-1], dataset[:, -1].astype(int)

    @staticmethod
    def two_classes_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        data_x_c1 = dataset[dataset[:, -1] == 0][:, :-1]
        data_x_c2 = dataset[dataset[:, -1] == 1][:, :-1]
        return data_x_c1, data_x_c2

    @staticmethod
    def two_classes_merge(data_x_c1: np.ndarray, data_x_c2: np.ndarray,
                          data_y: np.ndarray) -> np.array:
        data_x = np.append(data_x_c1, data_x_c2, axis=0)
        return np.append(data_x, data_y, axis=1)


class PCA:
    """ Principal Component Analysis. """
    means: np.ndarray
    basis_vector: np.ndarray

    def __init__(self) -> None:
        pass

    def fit(self, data: np.ndarray, max_dims: int = None, max_error: float = None):
        # Split features and class labels
        data_x, data_y = self.x_y_split(dataset=data)
        if max_dims:
            if max_dims > data_x.shape[1] - 1:
                raise Exception("Max dims should be no more than # of features -1!")
        elif not max_error:
            logger.warning("Neither of max_dims, max_error was given. Using max_dims=1")
            max_dims = 1

        # Calculate overall means
        self.means = np.expand_dims(np.mean(data_x, axis=0), axis=1).T
        # Calculate overall covariances
        covariances = np.cov(data_x.T)
        # Calculate lambdas and eigenvectors
        lambdas, eig_vectors = np.linalg.eig(covariances)
        # Calculate the basis vector based on the largest lambdas
        lambdas_sorted_idx = np.argsort(lambdas)[::-1]
        lambdas_sorted = lambdas[lambdas_sorted_idx]
        # If max_error is set, derive max_dims based on that
        if max_error:
            lambdas_sum = np.sum(lambdas_sorted)
            for n_dropped_dims in range(0, data_x.shape[1]):
                n_first_lambdas = lambdas_sorted[:n_dropped_dims + 1]
                pca_error = 1 - (np.sum(n_first_lambdas) / lambdas_sum)
                if pca_error <= max_error:
                    max_dims = n_dropped_dims+1
                    logger.info(f"For # dims={max_dims} error={pca_error} <= {max_error}")
                    break
        self.basis_vector = eig_vectors[:, lambdas_sorted_idx[:max_dims]]

    def transform(self, data: np.ndarray) -> np.array:
        data_x, data_y = self.x_y_split(dataset=data)
        data_x_proj = data_x @ self.basis_vector
        return np.append(data_x_proj, data_y[:, np.newaxis], axis=1)

    @staticmethod
    def x_y_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        return dataset[:, :-1], dataset[:, -1].astype(int)
