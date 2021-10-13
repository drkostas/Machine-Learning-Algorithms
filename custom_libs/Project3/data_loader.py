import numpy as np
import cv2
from typing import Tuple, List, Set
import os.path

from custom_libs import ColorizedLogger

logger = ColorizedLogger('Data Loader', 'cyan')


class DataLoader:
    """ Synth and Pima Data loading and preprocess Class. """
    active_datasets: List[str]
    all_datasets: Set[str] = {'synth', 'pima', 'flowers'}
    data_folder: str
    synth_tr: np.ndarray
    synth_te: np.ndarray
    pima_tr: np.ndarray
    pima_te: np.ndarray
    flowers: np.ndarray

    def __init__(self, datasets: List[str], data_folder: str = None):
        if len(self.all_datasets.intersection(datasets)) == 0:
            raise Exception(f"No supported datasets given. Options: {self.all_datasets}")
        self.active_datasets = datasets
        self.data_folder = data_folder
        self.load_datasets()

    def get_datasets(self) -> List[np.ndarray]:
        return_datasets = []
        if 'synth' in self.active_datasets:
            return_datasets.extend([self.synth_te, self.synth_tr])
        if 'pima' in self.active_datasets:
            return_datasets.extend([self.pima_tr, self.pima_te])
        if 'flowers' in self.active_datasets:
            return_datasets.extend([self.flowers])
        return return_datasets

    def load_datasets(self):
        if 'synth' in self.active_datasets:
            # Load and replace Yes/No with 0/1
            self.synth_te = np.genfromtxt(os.path.join(self.data_folder, 'synth.te'), autostrip=True)
            self.synth_tr = np.genfromtxt(os.path.join(self.data_folder, 'synth.tr'), autostrip=True)
        if 'pima' in self.active_datasets:
            pima_converter = {7: (lambda col: 0.0 if col.strip() == b"Yes" else 1.0)}
            self.pima_tr = np.genfromtxt(os.path.join(self.data_folder, 'pima.tr'),
                                         converters=pima_converter, autostrip=True)
            self.pima_te = np.genfromtxt(os.path.join(self.data_folder, 'pima.te'),
                                         converters=pima_converter, autostrip=True)
        if 'flowers' in self.active_datasets:
            # IT LOADS IT IT BGR!
            self.flowers = cv2.imread(os.path.join(self.data_folder, "flowersm.ppm"))

    def normalize_pima(self, print_statistics: bool = False) -> None:
        if 'pima' in self.active_datasets:
            # Normalize Pima
            pima_tr_orig = np.copy(self.pima_tr)
            pima_te_orig = np.copy(self.pima_te)
            pima_means = pima_tr_orig[:, :7].mean(axis=0)
            pima_max = pima_tr_orig[:, :7].max(axis=0)
            pima_min = pima_tr_orig[:, :7].min(axis=0)
            self.pima_tr[:, :7] = (pima_tr_orig[:, :7] - pima_means) / (pima_max-pima_min)
            self.pima_te[:, :7] = (pima_te_orig[:, :7] - pima_means) / (pima_max-pima_min)
            if print_statistics:
                self._print_statistics(self.pima_tr, "pima_tr")
                self._print_statistics(self.pima_te, "pima_te")
        else:
            logger.warning("Pima hasn't been loaded. Skipping..")

    def standarize_pima(self, print_statistics: bool = False) -> None:
        if 'pima' in self.active_datasets:
            # Normalize Pima
            pima_tr_orig = np.copy(self.pima_tr)
            pima_te_orig = np.copy(self.pima_te)
            pima_means = pima_tr_orig[:, :7].mean(axis=0)
            pima_stds = pima_tr_orig[:, :7].std(axis=0)
            self.pima_tr[:, :7] = (pima_tr_orig[:, :7] - pima_means) / pima_stds
            self.pima_te[:, :7] = (pima_te_orig[:, :7] - pima_means) / pima_stds
            if print_statistics:
                self._print_statistics(self.pima_tr, "pima_tr")
                self._print_statistics(self.pima_te, "pima_te")
        else:
            logger.warning("Pima hasn't been loaded. Skipping..")

    def print_statistics(self) -> None:
        if 'synth' in self.active_datasets:
            self._print_statistics(self.synth_tr, "synth_tr")
            self._print_statistics(self.synth_tr, "synth_tr")
        if 'pima' in self.active_datasets:
            self._print_statistics(self.pima_tr, "pima_tr")
            self._print_statistics(self.pima_te, "pima_te")
        if 'flowers' in self.active_datasets:
            self._print_statistics(self.flowers, "flowers")

    @staticmethod
    def _print_statistics(np_arr: np.array, var_name: str) -> None:
        logger.info(f"-- {var_name} --")
        logger.info(f"\tShape: {np_arr.shape}")
        logger.info(f"\tType: {np_arr.dtype}")
        logger.info(f"\tMean:")
        logger.info(f"\t\t{np_arr.mean(axis=0)[:4]} (..)")
        logger.info(f"\tMax:")
        logger.info(f"\t\t{np_arr.max(axis=0)[:4]} (..)")
        logger.info(f"\tMin:")
        logger.info(f"\t\t{np_arr.min(axis=0)[:4]} (..)")
        logger.info(f"\tHead:")
        logger.info(f"\t\t{np_arr[0, :4]} (..)")
