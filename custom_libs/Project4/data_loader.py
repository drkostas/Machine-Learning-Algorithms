import numpy as np
import cv2
from urllib import request
import gzip
import pickle
from typing import *
import os.path

from custom_libs import ColorizedLogger

logger = ColorizedLogger('Data Loader', 'cyan')


class DataLoader:
    """ Synth and Pima Data loading and preprocess Class. """
    active_datasets: List[str]
    all_datasets: Set[str] = {'synth', 'pima', 'flowers', 'xor', 'mnist'}
    data_folder: str
    synth_tr: np.ndarray
    synth_te: np.ndarray
    pima_tr: np.ndarray
    pima_te: np.ndarray
    flowers: np.ndarray
    xor: np.ndarray
    mnist: List[np.ndarray]

    def __init__(self, datasets: List[str], data_folder: str = None):
        if len(self.all_datasets.intersection(datasets)) == 0:
            raise Exception(f"No supported datasets given. Options: {self.all_datasets}")
        self.active_datasets = datasets
        self.data_folder = data_folder
        self.load_datasets()

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
            # IT LOADS IT IN BGR!
            self.flowers = cv2.imread(os.path.join(self.data_folder, "flowersm.ppm"))
        if 'xor' in self.active_datasets:
            self.xor = np.array(((0, 0, 0),
                                 (0, 1, 1),
                                 (1, 0, 1),
                                 (1, 1, 0)))
        if 'mnist' in self.active_datasets:
            mnist_loader = MnistDownloader()
            self.mnist = mnist_loader.load()
            self.mnist[1] = self.mnist[1][:, np.newaxis]
            self.mnist[3] = self.mnist[1][:, np.newaxis]

    def normalize_pima(self, print_statistics: bool = False) -> None:
        if 'pima' in self.active_datasets:
            # Normalize Pima
            pima_tr_orig = np.copy(self.pima_tr)
            pima_te_orig = np.copy(self.pima_te)
            pima_means = pima_tr_orig[:, :7].mean(axis=0)
            pima_max = pima_tr_orig[:, :7].max(axis=0)
            pima_min = pima_tr_orig[:, :7].min(axis=0)
            self.pima_tr[:, :7] = (pima_tr_orig[:, :7] - pima_means) / (pima_max - pima_min)
            self.pima_te[:, :7] = (pima_te_orig[:, :7] - pima_means) / (pima_max - pima_min)
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

    def get_datasets(self) -> Dict[str, List]:
        return_datasets = {}
        if 'synth' in self.active_datasets:
            return_datasets['synth'] = [self.synth_te, self.synth_tr]
        if 'pima' in self.active_datasets:
            return_datasets['pima'] = [self.pima_tr, self.pima_te]
        if 'flowers' in self.active_datasets:
            return_datasets['flowers'] = self.flowers
        if 'xor' in self.active_datasets:
            return_datasets['xor'] = self.xor
        if 'mnist' in self.active_datasets:
            return_datasets['mnist'] = self.mnist
        return return_datasets

    def print_statistics(self) -> None:
        if 'synth' in self.active_datasets:
            self._print_statistics(self.synth_tr, "synth_tr")
            self._print_statistics(self.synth_tr, "synth_tr")
        if 'pima' in self.active_datasets:
            self._print_statistics(self.pima_tr, "pima_tr")
            self._print_statistics(self.pima_te, "pima_te")
        if 'flowers' in self.active_datasets:
            self._print_statistics(self.flowers, "flowers")
        if 'xor' in self.active_datasets:
            self._print_statistics(self.xor, "xor")
        if 'mnist' in self.active_datasets:
            components = ('mnist_tr_x', 'mnist_tr_y', 'mnist_te_x', 'mnist_te_y')
            for name, data in zip(components, self.mnist):
                self._print_statistics(data, name)

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


class MnistDownloader:
    filename = [
        ["training_images", "train-images-idx3-ubyte.gz"],
        ["test_images", "t10k-images-idx3-ubyte.gz"],
        ["training_labels", "train-labels-idx1-ubyte.gz"],
        ["test_labels", "t10k-labels-idx1-ubyte.gz"]
    ]

    base_url = "http://yann.lecun.com/exdb/mnist/"
    pkl_path = "data/mnist.pkl"

    def __init__(self):
        if not os.path.exists(self.pkl_path):
            self.download_mnist()
            self.save_mnist()

    def download_mnist(self):
        for name in self.filename:
            logger.info(f"Downloading {name[1]} ...")
            request.urlretrieve(self.base_url + name[1], name[1])
        logger.info("Download complete.")

    def save_mnist(self):
        mnist = {}
        for name in self.filename[:2]:
            with gzip.open(name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
        for name in self.filename[-2:]:
            with gzip.open(name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(mnist, f)
        print("Save complete.")

    def load(self) -> List:
        with open(self.pkl_path, 'rb') as f:
            mnist = pickle.load(f)
        return [mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist[
            "test_labels"]]
