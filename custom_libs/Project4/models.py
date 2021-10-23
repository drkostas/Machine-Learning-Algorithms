import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from time import time
from typing import *
from custom_libs import ColorizedLogger, timeit

logger = ColorizedLogger('Project4 Models', 'green')


class NN:
    """ Neural Network Model. """

    def __init__(self) -> None:
        pass

    @staticmethod
    def x_y_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        return dataset[:, :-1], dataset[:, -1].astype(int)

    @staticmethod
    def two_classes_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        data_x_c1_idx = dataset[:, -1] == 0
        data_x_c1 = dataset[data_x_c1_idx][:, :-1]
        data_x_c2_idx = dataset[:, -1] == 1
        data_x_c2 = dataset[data_x_c2_idx][:, :-1]
        return data_x_c1, data_x_c2

    def fit(self, data: np.ndarray):
        pass

    def transform(self, data: np.ndarray) -> np.array:
        pass
