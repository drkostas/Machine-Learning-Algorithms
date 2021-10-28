import numpy as np
from typing import *
from custom_libs import ColorizedLogger, timeit

logger = ColorizedLogger('Project4 Models', 'green')


class MultiLayerPerceptron:
    """ Multi Layer Perceptron Model. """
    n_layers: int
    units: List[int]
    biases: List[np.ndarray]
    weights: List[np.ndarray]
    activation: List[Union[None, Callable]]
    activation_derivative: List[Union[None, Callable]]

    def __init__(self, units: List[int], activations: List[str], seed: int = None) -> None:
        """
            g = activation function
            z = w.T @ a_previous + b
            a = g(z)
        """
        if seed:
            np.random.seed(seed)
        self.units = units
        self.n_layers = len(self.units)
        activations = ['identity' if activation_str is None else activation_str
                       for activation_str in activations]
        self.activation = [getattr(self, activation_str)
                           for activation_str in activations]
        self.activation_derivative = [getattr(self, f"{activation_str}_derivative")
                                      for activation_str in activations]
        self.initialize_weights()

    def initialize_weights(self):
        self.biases = [np.random.randn(y, 1) for y in self.units[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.units[:-1], self.units[1:])]

        # self.biases = [np.ones((y, 1)) for y in self.units[1:]]
        # self.weights = [np.zeros((y, x)) for x, y in zip(self.units[:-1], self.units[1:])]
        logger.info(f"Shapes of biases: {[bias.shape for bias in self.biases]}")
        logger.info(f"Shapes of weights: {[weights.shape for weights in self.weights]}")

    def train(self, train: np.ndarray = None, train_x: np.ndarray = None, train_y: np.ndarray = None,
              batch_size: int = 1, lr: float = 0.01, max_epochs: int = 1000, shuffle: bool = False,
              regularization: str = None, debug: Dict = None):

        if not debug:
            debug = {'top': 0, 'ff': 0, 'bp': 0, 'w': 0}
        # Ensure train dataset was provided correctly
        if train is None and (train_x is None or train_y is None):
            raise Exception("You should either set train or (train_x and train_y)!")
        # Split or concat x and y
        if train is not None:
            train_x, train_y = self.x_y_split(train)
        else:
            train = np.concatenate([train_x, train_y], axis=1)
        # Check input layer shape
        if train_x.shape[1] != self.units[0]:
            raise Exception("train.shape[1] is not matching units[0]!")

        accuracies = []
        losses = []

        # --- Train Loop --- #
        for epoch in range(1, max_epochs + 1):
            if debug['top'] > 2 or (debug['top'] > 1 and epoch % 1 == 0) or \
                    (debug['top'] > 0 and epoch % 100 == 0):
                logger.info(f"Epoch: {epoch}", color="red")
                show_batch = True
            else:
                show_batch = False
            # Shuffle
            if shuffle:
                np.random.shuffle(train)
            train_batches = [train[k:k + batch_size] for k in range(0, train.shape[0], batch_size)]
            # Run mini-batches
            for batch_ind, train_batch in enumerate(train_batches):
                if debug['top'] > 2 and show_batch:
                    logger.info(f"  Batch: {batch_ind}", color='yellow')
                self.run_batch(data_batch=train_batch, lr=lr, debug=debug)
            accuracy = self.accuracy(train_x, train_y)
            loss = self.total_cost(train_x, train_y)
            accuracies.append(accuracy / train.shape[0])
            losses.append(loss)
            if show_batch:
                logger.info(f"  Loss: {loss[0]:.5f}")
                logger.info(
                    "  Accuracy on training data: {} / {}".format(accuracy, train.shape[0]))
        return accuracies, losses

    def run_batch(self, data_batch: np.ndarray, lr: float, debug: Dict):
        batch_x, batch_y = self.x_y_split(data_batch)
        for batch_iter, (row_x, row_y) in enumerate(zip(batch_x, batch_y)):
            row_x, row_y = row_x[np.newaxis, :], row_y[:, np.newaxis]
            z, a = self.feed_forward(row_x, debug)
            dw_, db_ = self.back_propagation(row_y, z, a, debug)
            if batch_iter == 0:
                dw = dw_
                db = db_
            else:
                dw = list(map(np.add, dw, dw_))
                db = list(map(np.add, db, db_))

        self.update_weights_and_biases(dw, db, lr, batch_iter + 1, debug)

    def feed_forward(self, batch_x: np.ndarray, debug: Dict = None) -> \
            Tuple[List[np.ndarray], List[np.ndarray]]:
        if not debug:
            debug = {'ff': 0}
        z_ = batch_x.T
        z = [z_]
        a_ = z_
        a = [a_]
        for l_ind, layer_units in enumerate(self.units[1:]):
            z_ = self.weights[l_ind] @ a_ + self.biases[l_ind]  # a_ -> a_previous
            z.append(z_)
            a_ = self.activation[l_ind](z_)
            a.append(a_)
            if debug['ff'] > 2:
                if l_ind == 0:
                    logger.info("    Feed Forward", color="cyan")
                logger.info(f"      Layer: {l_ind}, units: {layer_units}", color="magenta")
                logger.info(f"        z{z_.T} = w[{l_ind}]{self.weights[l_ind]} @ a_ + "
                            f"b[{l_ind}]{self.biases[l_ind].T}")
                logger.info(f"        a{a_.T} = g[{l_ind}](z{z_.T})")

        return z, a

    def back_propagation(self, batch_y: np.ndarray, z: List[np.ndarray], a: List[np.ndarray],
                         debug: Dict = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if not debug:
            debug = {'bp': 0}
        db = []
        dw = []
        # Calculate backprop input which is da of last layer
        da = self.cost_derivative(a[-1], batch_y)
        for l_ind, layer_units in list(enumerate(self.units))[-1:0:-1]:  # layers: last->2nd
            g_prime = self.activation_derivative[l_ind - 1](z[l_ind])
            dz = da * g_prime
            db_ = dz
            dw_ = dz @ a[l_ind - 1].T
            da = self.weights[l_ind - 1].T @ dz  # To be used in the next iteration (previous layer)
            db.append(db_)
            dw.append(dw_)
            if debug['bp'] > 2:
                if layer_units == self.units[-1]:
                    logger.info("    Back Propagation", color="cyan")
                logger.info(f"      Layer: {l_ind}, units: {layer_units}", color="magenta")
                logger.info(f"        g_prime{g_prime.shape} = activation_derivative[{l_ind - 1}]"
                            f"(z[{l_ind}]{z[l_ind].shape})"
                            f"{self.activation_derivative[l_ind - 1](z[l_ind]).shape}")
                logger.info(f"        dz{dz.shape} = da{da.shape} * g_prime{g_prime.shape}")
                logger.info(f"        db{db_.shape} = dz{dz.shape}")
                logger.info(f"        dw = dz{dz.shape} @ a[{l_ind - 1}]{a[l_ind - 1].shape}")
                logger.info(f"        da{da.shape} = self.weights[{l_ind - 1}].T"
                            f"{self.weights[l_ind - 1].T.shape} @ dz{dz.shape}")

        dw.reverse()
        db.reverse()
        return dw, db

    def update_weights_and_biases(self, dw: List[np.ndarray], db: List[np.ndarray],
                                  lr: float, batch_size: int, debug: Dict = None) -> None:
        if not debug:
            debug = {'w': 0}
        for l_ind, layer_units in enumerate(self.units[:-1]):
            self.weights[l_ind] -= (lr / batch_size) * dw[l_ind]
            self.biases[l_ind] -= (lr / batch_size) * db[l_ind]

            if debug['w'] > 2:
                if l_ind == 0:
                    logger.info("    Update Weights", color="cyan")
                logger.info(f"      Layer: {l_ind}, units: {layer_units}", color="magenta")
                logger.info(f"        w({self.weights[l_ind].shape}) -= "
                            f"({lr}/{batch_size}) * dw({dw[l_ind].shape}")
                logger.info(f"        b({self.weights[l_ind].shape}) -= "
                            f"({lr}/{batch_size}) * db({db[l_ind].shape}")
    @staticmethod
    def identity(z):
        return z

    identity_derivative = identity

    @staticmethod
    def cost_derivative(output_z, y):
        return output_z - y

    @staticmethod
    def sigmoid(z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def sigmoid_derivative(cls, z):
        """Derivative of the sigmoid function."""
        return cls.sigmoid(z) * (1 - cls.sigmoid(z))

    def predict(self, x: Iterable[np.ndarray]) -> \
            Tuple[Iterable[int], Iterable[np.ndarray]]:
        y_predicted = []
        y_raw_predictions = []
        for x_row in x:
            x_row = x_row[np.newaxis, :]
            z, a = self.feed_forward(x_row)
            prediction_raw = a[-1]
            y_raw_predictions.append(prediction_raw)
            prediction = self.classify(prediction_raw)
            # logger.info(f"weight: {prediction_raw}, y_pred: {prediction}")
            y_predicted.append(prediction)
        return y_predicted, y_raw_predictions

    def accuracy(self, data_x: np.ndarray, data_y: np.ndarray) -> int:
        # logger.nl()
        predictions, _ = self.predict(data_x)
        result_accuracy = sum(int(pred == true) for (pred, true) in zip(predictions, data_y))
        return result_accuracy

    def total_cost(self, data_x: np.ndarray, data_y: np.ndarray):
        predictions, predictions_raw = self.predict(data_x)
        sum_cost = 0.0
        for ind, prediction_raw in enumerate(predictions_raw):
            current_y = int(data_y[ind])
            try:
                prediction_raw = prediction_raw[current_y]
            except Exception as e:
                print("current_y: ", current_y)
                print("prediction_raw: ", prediction_raw)
                raise e
            sum_cost += current_y * np.log(1e-15 + prediction_raw)
        mean_cost = -1.0/len(data_y) * sum_cost
        return mean_cost

    @staticmethod
    def classify(y: np.ndarray) -> int:
        return y.argmax()

    @staticmethod
    def x_y_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        return dataset[:, :-1], dataset[:, -1][:, np.newaxis].astype(int)

    @staticmethod
    def two_classes_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        data_x_c1_idx = dataset[:, -1] == 0
        data_x_c1 = dataset[data_x_c1_idx][:, :-1]
        data_x_c2_idx = dataset[:, -1] == 1
        data_x_c2 = dataset[data_x_c2_idx][:, :-1]
        return data_x_c1, data_x_c2
