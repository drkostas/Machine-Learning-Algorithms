import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


class Plotter:
    pima_tr: np.ndarray
    pima_te: np.ndarray

    def __init__(self, pima_tr: np.ndarray, pima_te: np.ndarray):
        self.pima_tr = pima_tr
        self.pima_te = pima_te
