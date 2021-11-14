import numpy as np
import cv2
from urllib import request
import gzip
import pickle
from typing import *
import os.path
from sklearn.model_selection import train_test_split

from custom_libs import ColorizedLogger

logger = ColorizedLogger('Data Loader', 'cyan')


class DataLoader:
    """ Datasets loading and preprocess Class. """
    active_datasets: List[str]
    all_datasets: Set[str] = {'synth', 'pima', 'flowers', 'xor', 'mnist'}
    data_folder: str
    synth_tr: np.ndarray
    synth_te: np.ndarray
    pima_tr: np.ndarray
    pima_te: np.ndarray
    pima_val: np.ndarray
    flowers: np.ndarray
    xor: np.ndarray
    mnist_tr: np.ndarray
    mnist_te: np.ndarray
    mnist_val: np.ndarray

    def __init__(self, datasets: List[str], data_folder: str = None, seed: int = None):
        if seed:
            np.seed(seed)
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
            pima_tr = np.genfromtxt(os.path.join(self.data_folder, 'pima.tr'),
                                         converters=pima_converter, autostrip=True)
            self.pima_tr, self.pima_val = train_test_split(pima_tr, test_size=0.3)
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
            mnist = mnist_loader.load()
            if len(mnist) == 4:
                self.mnist_tr = np.concatenate([mnist[0], mnist[1][:, np.newaxis]], axis=1)
                mnist_te = np.concatenate([mnist[2], mnist[3][:, np.newaxis]], axis=1)
                self.mnist_te, self.mnist_val = train_test_split(mnist_te, test_size=0.5)
            else:
                mnist_tr, mnist_te, mnist_val = mnist
                self.mnist_tr = np.concatenate([mnist_tr[0], mnist_tr[1][:, np.newaxis]], axis=1)
                self.mnist_te = np.concatenate([mnist_te[0], mnist_te[1][:, np.newaxis]], axis=1)
                self.mnist_val = np.concatenate([mnist_val[0], mnist_val[1][:, np.newaxis]], axis=1)

    @staticmethod
    def normalize_(train, test=None, val_set=None, epsilon: float = 1e-100):
        train_ = np.copy(train).astype(np.float64)
        means_ = train_[:, :-1].mean(axis=0)
        max_ = train_[:, :-1].max(axis=0)
        min_ = train_[:, :-1].min(axis=0)
        denominator = (max_ - min_ + epsilon)
        train_[:, :-1] = (train_[:, :-1] - means_) / denominator
        return_datasets = [train_]
        if test is not None:
            test_ = np.copy(test).astype(np.float64)
            test_[:, :-1] = (test_[:, :-1] - means_) / denominator
            return_datasets.append(test_)
        if val_set is not None:
            val_set_ = np.copy(val_set).astype(np.float64)
            val_set_[:, :-1] = (val_set_[:, :-1] - means_) / denominator
            return_datasets.append(val_set_)
        return tuple(return_datasets)

    @staticmethod
    def standarize_(train, test=None, val_set=None, epsilon: float = 1e-100):
        train_ = np.copy(train).astype(np.float64)
        means_ = train_[:, :-1].mean(axis=0)
        std_ = train_[:, :-1].std(axis=0) + epsilon
        train_[:, :-1] = (train_[:, :-1] - means_) / std_
        return_datasets = [train_]
        if test is not None:
            test_ = np.copy(test).astype(np.float64)
            test_[:, :-1] = (test_[:, :-1] - means_) / std_
            return_datasets.append(test_)
        if val_set is not None:
            val_set_ = np.copy(val_set).astype(np.float64)
            val_set_[:, :-1] = (val_set_[:, :-1] - means_) / std_
            return_datasets.append(val_set_)
        return tuple(return_datasets)

    def normalize(self, dataset: str) -> None:
        if dataset == 'synth' and dataset in self.active_datasets:
            self.synth_te, self.synth_tr = self.normalize_(self.synth_te, self.synth_tr)
        elif dataset == 'pima' and dataset in self.active_datasets:
            self.pima_tr, self.pima_te, self.pima_val = \
                self.normalize_(self.pima_tr, self.pima_te, self.pima_val)
        elif dataset == 'flowers' and dataset in self.active_datasets:
            self.flowers = self.normalize_(self.flowers)[0]
        elif dataset == 'xor' and dataset in self.active_datasets:
            self.xor = self.normalize_(self.xor)[0]
        elif dataset == 'mnist' and dataset in self.active_datasets:
            self.mnist_tr, self.mnist_te, self.mnist_val = self.normalize_(self.mnist_tr,
                                                                           self.mnist_te,
                                                                           self.mnist_val)
        else:
            logger.warning(f"{dataset} hasn't been loaded or not supported.")

    def standarize(self, dataset: str) -> None:
        if dataset == 'synth' and dataset in self.active_datasets:
            self.synth_te, self.synth_tr = self.standarize_(self.synth_te, self.synth_tr)
        elif dataset == 'pima' and dataset in self.active_datasets:
            self.pima_tr, self.pima_te, self.pima_val = \
                self.standarize_(self.pima_tr, self.pima_te, self.pima_val)
        elif dataset == 'flowers' and dataset in self.active_datasets:
            self.flowers = self.standarize_(self.flowers)[0]
        elif dataset == 'xor' and dataset in self.active_datasets:
            self.xor = self.standarize_(self.xor)[0]
        elif dataset == 'mnist' and dataset in self.active_datasets:
            self.mnist_tr, self.mnist_te, self.mnist_val = self.standarize_(self.mnist_tr,
                                                                            self.mnist_te,
                                                                            self.mnist_val)
        else:
            logger.warning(f"{dataset} hasn't been loaded or not supported.")

    @staticmethod
    def one_hot_encode_last(data):
        y = data[:, -1].copy().T.astype(int)
        y_one_hot = np.zeros((y.size, y.max() + 1))
        y_one_hot[np.arange(y.size), y] = 1
        return y_one_hot

    def get_datasets(self) -> Dict[str, List]:
        return_datasets = {}
        if 'synth' in self.active_datasets:
            return_datasets['synth'] = [self.synth_te, self.synth_tr]
        if 'pima' in self.active_datasets:
            return_datasets['pima'] = [self.pima_tr, self.pima_te, self.pima_val]
        if 'flowers' in self.active_datasets:
            return_datasets['flowers'] = self.flowers
        if 'xor' in self.active_datasets:
            return_datasets['xor'] = self.xor
        if 'mnist' in self.active_datasets:
            return_datasets['mnist'] = [self.mnist_tr, self.mnist_te, self.mnist_val]
        return return_datasets

    def print_statistics(self) -> None:
        if 'synth' in self.active_datasets:
            self._print_statistics(self.synth_tr, "synth_tr")
            self._print_statistics(self.synth_tr, "synth_tr")
        if 'pima' in self.active_datasets:
            self._print_statistics(self.pima_tr, "pima_tr")
            self._print_statistics(self.pima_te, "pima_te")
            self._print_statistics(self.pima_val, "pima_val")
        if 'flowers' in self.active_datasets:
            self._print_statistics(self.flowers, "flowers")
        if 'xor' in self.active_datasets:
            self._print_statistics(self.xor, "xor")
        if 'mnist' in self.active_datasets:
            self._print_statistics(self.mnist_tr, "mnist_tr")
            self._print_statistics(self.mnist_te, "mnist_te")
            self._print_statistics(self.mnist_val, "mnist_val")

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
        try:
            with open(self.pkl_path, 'rb') as f:
                mnist = pickle.load(f)
            datasets = [mnist["training_images"], mnist["training_labels"],
                        mnist["test_images"], mnist["test_labels"]]
        except UnicodeDecodeError as e:
            with open(self.pkl_path, 'rb') as f:
                mnist = pickle._Unpickler(f)
                mnist.encoding = 'latin1'
                datasets = list(mnist.load())
        return datasets
