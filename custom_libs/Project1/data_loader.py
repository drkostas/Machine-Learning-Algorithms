import numpy as np
from typing import Tuple


class DataLoader:
    """ Synth and Pima Data loading and preprocess Class. """
    synth_path: str
    pima_path: str
    synth_tr: np.ndarray
    synth_te: np.ndarray
    pima_tr: np.ndarray
    pima_te: np.ndarray

    def __init__(self, synth_path: str, pima_path: str):
        self.synth_path = synth_path
        self.pima_path = pima_path
        self.load_datasets()

    def get_datasets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.synth_te, self.synth_tr, self.pima_tr, self.pima_te

    def load_datasets(self):
        # Load and replace Yes/No with 0/1
        self.synth_te = np.genfromtxt('data/synth.te', autostrip=True)
        self.synth_tr = np.genfromtxt('data/synth.tr', autostrip=True)
        pima_converter = {7: (lambda col: 0.0 if col.strip() == b"Yes" else 1.0)}
        self.pima_tr = np.genfromtxt('data/pima.tr', converters=pima_converter, autostrip=True)
        self.pima_te = np.genfromtxt('data/pima.te', converters=pima_converter, autostrip=True)

    def normalize_pima(self, print_statistics: bool = False) -> None:
        # Preprocessing
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

    def print_statistics(self) -> None:
        self._print_statistics(self.synth_tr, "synth_tr")
        self._print_statistics(self.synth_tr, "synth_tr")
        self._print_statistics(self.pima_tr, "pima_tr")
        self._print_statistics(self.pima_te, "pima_te")

    @staticmethod
    def _print_statistics(np_arr: np.array, var_name: str) -> None:
        print(f"-- {var_name} --")
        print(f"\tShape: {np_arr.shape}")
        print(f"\tType: {np_arr.dtype}")
        print(f"\tMean:\n\t\t{np_arr.mean(axis=0)[:4]} (..)")
        print(f"\tMax:\n\t\t{np_arr.max(axis=0)[:4]} (..)")
        print(f"\tMin:\n\t\t{np_arr.min(axis=0)[:4]} (..)")
        print(f"\tHead:\n\t\t{np_arr[0, :4]} (..)")
