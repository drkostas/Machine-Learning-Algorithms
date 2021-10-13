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
        s1 = np.matmul(data_minus_means_c1, data_minus_means_c1.T)
        data_minus_means_c2 = (means_c2.T - data_x_c2).T
        s2 = np.matmul(data_minus_means_c2, data_minus_means_c2.T)
        # Calculate Sw, Sw inverse
        sw = s1 + s2
        sw_inv = np.linalg.inv(sw)
        # Calculate the difference of the two means (equivalent to Sb)
        class_means_diff = means_c1 - means_c2
        # Calculate the projection vector
        self.w = np.matmul(sw_inv, class_means_diff)

    def transform(self, data: np.ndarray) -> Tuple[np.array, np.array]:
        data_x_c1, data_x_c2 = self.two_classes_split(dataset=data)
        data_x_proj_c1 = np.matmul(data_x_c1, self.w)
        data_x_proj_c2 = np.matmul(data_x_c2, self.w)
        return data_x_proj_c1, data_x_proj_c2

    @staticmethod
    def x_y_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        return dataset[:, :-1], dataset[:, -1].astype(int)

    @staticmethod
    def two_classes_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        data_x_c1 = dataset[dataset[:, -1] == 0][:, :-1]
        data_x_c2 = dataset[dataset[:, -1] == 1][:, :-1]
        return data_x_c1, data_x_c2

class PCA:
    """ Principal Component Analysis. """
    means: np.ndarray
    basis_vector: np.ndarray

    def __init__(self) -> None:
        pass

    def fit(self, data: np.ndarray):
        data_x, data_y = self.x_y_split(dataset=data)
        # Calculate overall means
        self.means = np.expand_dims(np.mean(data_x, axis=0), axis=1).T
        # Calculate overall covariances
        covariances = np.cov(data_x.T)
        # Calculate lambdas and eigenvectors
        lambdas, eig_vectors = np.linalg.eig(covariances)
        # Calculate the basis vector based on the largest lambdas
        self.basis_vector = np.expand_dims(eig_vectors[:, np.argmax(lambdas)], axis=1)

    def transform(self, data: np.ndarray) -> np.array:
        data_x, data_y = self.x_y_split(dataset=data)
        data_x_proj = self.means + (np.matmul(self.basis_vector, self.basis_vector.T) @ (data_x-self.means).T).T
        return data_x_proj

    @staticmethod
    def x_y_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        return dataset[:, :-1], dataset[:, -1].astype(int)
