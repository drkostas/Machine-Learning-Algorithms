import numpy as np
from typing import *
from custom_libs import ColorizedLogger, timeit

logger = ColorizedLogger('Project4 Models', 'green')

np.seterr(all='warn')
np.seterr(all='raise')


class MultiLayerPerceptron:
    """ Multi Layer Perceptron Model. """
    n_layers: int
    units: List[int]
    biases: List[np.ndarray]
    weights: List[np.ndarray]
    activation: List[Union[None, Callable]]
    activation_derivative: List[Union[None, Callable]]
    loss_functions: List[Callable]
    loss_function_derivatives: List[Callable]

    def __init__(self, units: List[int], activations: List[str], loss_functions: Iterable[str],
                 seed: int = None) -> None:
        """
            g = activation function
            z = w.T @ a_previous + b
            a = g(z)
        """
        if seed:
            np.random.seed(seed)
        self.units = units
        logger.info(f"Units per Layer: {self.units}")
        self.n_layers = len(self.units)
        activations = ['linear' if activation_str is None else activation_str
                       for activation_str in activations]
        self.activation = [getattr(self, activation_str)
                           for activation_str in activations]
        self.activation_derivative = [getattr(self, f"{activation_str}_derivative")
                                      for activation_str in activations]
        self.loss_functions = [getattr(self, loss_function) for loss_function in loss_functions]
        self.loss_function_derivatives = [getattr(self, f"{loss_function}_derivative")
                                          for loss_function in loss_functions]
        self.initialize_weights()

    def initialize_weights(self):
        self.biases = [np.random.randn(y, 1) for y in self.units[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.units[:-1], self.units[1:])]
        # self.biases = [np.ones((y, 1)) for y in self.units[1:]]
        # self.weights = [np.zeros((y, x)) for x, y in zip(self.units[:-1], self.units[1:])]
        logger.info(f"Shapes of biases: {[bias.shape for bias in self.biases]}")
        logger.info(f"Shapes of weights: {[weights.shape for weights in self.weights]}")

    def train(self, data: np.ndarray, one_hot_y: np.ndarray,
              batch_size: int = 1, lr: float = 0.01, momentum: float = 0.0,
              max_epochs: int = 1000, early_stopping: Dict = None, shuffle: bool = False,
              regularization_param: float = 0.0, debug: Dict = None):
        # Set Default values
        if not debug:
            debug = {'epochs': 10 ** 10, 'batches': 10 ** 10, 'ff': False, 'bp': False, 'w': False}
        # Lists to gather accuracies and losses
        accuracies = []
        losses = []
        # --- Train Loop --- #
        data_x, _ = self.x_y_split(data)
        for epoch in range(1, max_epochs + 1):
            if epoch % debug['epochs'] == 0:
                logger.info(f"Epoch: {epoch}", color="red")
                show_epoch = True
            else:
                show_epoch = False
            # Shuffle
            if shuffle:
                shuffle_idx = np.random.permutation(data_x.shape[0])
                data_x = data_x[shuffle_idx, :]
                one_hot_y = one_hot_y[shuffle_idx, :]
            # Create Mini-Batches
            train_batches = [(data_x[k:k + batch_size], one_hot_y[k:k + batch_size])
                             for k in range(0, data_x.shape[0], batch_size)]
            # Run mini-batches
            for batch_ind, (x_batch, one_hot_y_batch) in enumerate(train_batches):
                batch_ind += 1
                if show_epoch and batch_ind % debug['batches'] == 0:
                    logger.info(f"  Batch: {batch_ind}", color='yellow')
                self.run_batch(batch_x=x_batch, batch_y=one_hot_y_batch, lr=lr, momentum=momentum,
                               regularization_param=regularization_param, debug=debug)
                # Calculate Batch Accuracy and Losses
                if show_epoch and batch_ind % debug['batches'] == 0:
                    accuracy = self.accuracy(data_x, one_hot_y, debug)
                    batch_losses = self.total_loss(data_x, one_hot_y, regularization_param, debug)
                    self.print_stats(batch_losses, accuracy, data_x.shape[0], '    ')

            # Gather Results
            accuracy = self.accuracy(data_x, one_hot_y, debug)
            epoch_losses = self.total_loss(data_x, one_hot_y, regularization_param, debug)
            accuracies.append(accuracy / data_x.shape[0])
            losses.append(epoch_losses)
            # Calculate Epoch Accuracy and Losses
            if show_epoch:
                self.print_stats(epoch_losses, accuracy, data_x.shape[0], '  ')
            if early_stopping:
                if 'accuracy' in early_stopping and epoch > early_stopping['wait']:
                    if accuracies[-1]-accuracies[-2] < early_stopping['accuracy']:
                        logger.info(f"Early stopping (acc): {accuracies[-1]}-{accuracies[-2]} = "
                                    f"{(accuracies[-1] - accuracies[-2])} < "
                                    f"{early_stopping['accuracy']}", color='yellow')
                        break
                if 'loss' in early_stopping and epoch > early_stopping['wait']:
                    if losses[-1][0][1]-losses[-2][0][1] < early_stopping['loss']:
                        print(losses[-1][0][1], losses[-2][0][1])

                        logger.info(f"Early stopping (loss): "
                                    f"{losses[-1][0][1]:5f}-{losses[-2][0][1]:5f} = "
                                    f"{(losses[-1][0][1] - losses[-2][0][1]):5f} < "
                                    f"{early_stopping['loss']}", color='yellow')
                        break

        logger.info(f"Finished after {epoch} epochs", color='red')
        self.print_stats(epoch_losses, accuracy, data_x.shape[0], '')

        return accuracies, losses

    @staticmethod
    def print_stats(losses, accuracy, size, padding):
        for loss_type, loss in losses:
            logger.info(f"{padding}{loss_type} Loss: {loss:.5f}")
        logger.info(f"{padding}Accuracy on training data: {accuracy}/{size}")

    def run_batch(self, batch_x: np.ndarray, batch_y: np.ndarray, lr: float,
                  momentum: float, regularization_param: float, debug: Dict):
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

        self.update_weights_and_biases(dw, db, lr, momentum, batch_iter + 1,
                                       regularization_param, debug)

    def feed_forward(self, batch_x: np.ndarray, debug: Dict = None) -> \
            Tuple[List[np.ndarray], List[np.ndarray]]:
        if debug is None:
            debug = {'ff': False}
        z_ = batch_x.T
        z = [z_]
        a_ = z_
        a = [a_]
        for l_ind, layer_units in enumerate(self.units[1:]):
            z_ = self.weights[l_ind] @ a_ + self.biases[l_ind]  # a_ -> a_previous
            z.append(z_)
            a_ = self.activation[l_ind](z_)
            a.append(a_)
            if debug['ff']:
                if l_ind == 0:
                    logger.info("    Feed Forward", color="cyan")
                logger.info(f"      Layer: {l_ind}, units: {layer_units}", color="magenta")
                logger.info(f"        z{z_.T} = w[{l_ind}]{self.weights[l_ind]} @ a_ + "
                            f"b[{l_ind}]{self.biases[l_ind].T}")
                logger.info(f"        a{a_.T} = g[{l_ind}](z{z_.T})")

        return z, a

    def back_propagation(self, batch_y: np.ndarray, z: List[np.ndarray], a: List[np.ndarray],
                         debug: Dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        db = []
        dw = []
        # Calculate back propagation input which is da of last layer
        da = self.loss_function_derivatives[0](z[-1], a[-1], batch_y)
        for l_ind, layer_units in list(enumerate(self.units))[-1:0:-1]:  # layers: last->2nd
            g_prime = self.activation_derivative[l_ind - 1](z[l_ind])
            try:
                dz = da * g_prime
            except Exception as e:
                print("l_ind: ", l_ind)
                print("layer_units: ", layer_units)
                print("da: ", da)
                print("g_prime: ", g_prime)
                raise e
            db_ = dz
            dw_ = dz @ a[l_ind - 1].T
            da = self.weights[l_ind - 1].T @ dz  # To be used in the next iteration (previous layer)
            db.append(db_)
            dw.append(dw_)
            if debug['bp']:
                if layer_units == self.units[-1]:
                    logger.info("    Back Propagation", color="cyan")
                logger.info(f"      Layer: {l_ind}, units: {layer_units}", color="magenta")
                logger.info(f"        g_prime{g_prime.shape} = activation_derivative[{l_ind - 1}]"
                            f"(z[{l_ind}]{z[l_ind].shape})"
                            f"{self.activation_derivative[l_ind - 1](z[l_ind]).shape} =\n"
                            f"\t\t\t\t\t\t\t{g_prime.T}")
                logger.info(f"        dz{dz.shape} = da{da.shape} * g_prime{g_prime.shape}")
                logger.info(f"        db{db_.shape} = dz{dz.shape}")
                logger.info(f"        dw = dz{dz.shape} @ a[{l_ind - 1}]{a[l_ind - 1].shape}")
                logger.info(f"        da{da.shape} = self.weights[{l_ind - 1}].T"
                            f"{self.weights[l_ind - 1].T.shape} @ dz{dz.shape} = \n"
                            f"\t\t\t\t\t\t\t{da.T}")

        dw.reverse()
        db.reverse()
        return dw, db

    def update_weights_and_biases(self, dw: List[np.ndarray], db: List[np.ndarray],
                                  lr: float, momentum: float, batch_size: int,
                                  regularization_param: float, debug: Dict) -> None:
        for l_ind, layer_units in enumerate(self.units[:-1]):
            # self.weights[l_ind] -= (lr / batch_size) * dw[l_ind]
            self.weights[l_ind] = (1 - lr * (regularization_param / batch_size)) * self.weights[
                l_ind] - (lr / batch_size) * dw[l_ind] + momentum * self.weights[l_ind]
            self.biases[l_ind] -= (lr / batch_size) * db[l_ind]

            if debug['w']:
                if l_ind == 0:
                    logger.info("    Update Weights", color="cyan")
                logger.info(f"      Layer: {l_ind}, units: {layer_units}", color="magenta")
                logger.info(f"        w({self.weights[l_ind].shape}) -= "
                            f"({lr}/{batch_size}) * dw({dw[l_ind].shape}")
                logger.info(f"        b({self.weights[l_ind].shape}) -= "
                            f"({lr}/{batch_size}) * db({db[l_ind].shape}")

    @staticmethod
    def linear(z):
        return z

    linear_derivative = linear

    @staticmethod
    def sigmoid(z):
        """The sigmoid function."""
        z = np.clip(z, -500, 500)  # Handle np.exp overflow
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    @classmethod
    def sigmoid_derivative(cls, a):
        """Derivative of the sigmoid function."""
        return cls.sigmoid(a) * (1 - cls.sigmoid(a))

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z).astype(z.dtype)

    @staticmethod
    def relu_derivative(a):
        return (a > 0).astype(a.dtype)

    @staticmethod
    def tanh(z):
        """ Should use different loss. """
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(a):
        """ Should use different loss. """
        return 1 - a ** 2

    @staticmethod
    def softmax(z):
        # y = np.exp(z - np.max(z))
        # a = y / np.sum(np.exp(z))
        from scipy.special import softmax
        a = softmax(z)
        return a

    softmax_derivative = sigmoid_derivative

    @staticmethod
    def classify(y: np.ndarray) -> np.ndarray:
        total = y.shape[0]
        prediction = np.zeros(total)
        prediction[y.argmax()] = prediction[y.argmax()] = 1
        return prediction

    def predict(self, x: Iterable[np.ndarray], debug: bool = False) -> \
            Tuple[List[np.ndarray], List[np.ndarray]]:
        y_predicted = []
        y_raw_predictions = []
        for x_row in x:
            if debug:
                logger.info(f"  x_row: {x_row[:20].T}", color='white')
            x_row = x_row[np.newaxis, :]
            z, a = self.feed_forward(x_row)
            prediction_raw = a[-1]
            prediction = self.classify(prediction_raw)
            if debug:
                logger.info(f"  prediction_raw: {prediction_raw.T}")
                logger.info(f"  prediction: {prediction}")
            y_raw_predictions.append(prediction_raw)
            y_predicted.append(prediction)
        return y_predicted, y_raw_predictions

    def accuracy(self, data_x: np.ndarray, data_y: np.ndarray,
                 debug: Dict) -> int:
        if debug['metrics']:
            logger.nl()
            logger.info('Accuracy', color='cyan')
        predictions, _ = self.predict(data_x, debug=debug['metrics'])
        result_accuracy = sum(int(np.array_equal(pred.astype(int), true.astype(int)))
                              for (pred, true) in zip(predictions, data_y))
        if debug['metrics']:
            logger.info(f'result_accuracy: {result_accuracy}')
        return result_accuracy

    def total_loss(self, data_x: np.ndarray, data_y: np.ndarray, regularization_param: float,
                   debug: Dict) -> List[Tuple[str, float]]:
        if debug['metrics']:
            logger.nl()
            logger.info('Total Loss', color='cyan')
        predictions, predictions_raw = self.predict(data_x, debug['metrics'])
        mean_costs = [0.0 for _ in range(len(self.loss_functions))]
        for ind, prediction_raw in enumerate(predictions_raw):
            current_y = data_y[ind]
            for loss_ind, loss_func in enumerate(self.loss_functions):
                mean_costs[loss_ind] += loss_func(prediction_raw, current_y)/len(predictions_raw)
                mean_costs[loss_ind] += 0.5 * (regularization_param / len(predictions_raw)) * sum(
                    np.linalg.norm(w) ** 2
                    for w in self.weights)
            if debug['metrics']:
                logger.info(f'ind: {ind}, prediction_raw: {prediction_raw.T}, current_y: {current_y}')
        costs_with_names = []
        for loss_ind, loss_func in enumerate(self.loss_functions):
            costs_with_names.append((loss_func.__name__, 1.0 / len(data_y) * mean_costs[loss_ind]))
        if debug['metrics']:
            logger.info(f'Mean Costs: {mean_costs}')
        return costs_with_names

    @staticmethod
    def cross_entropy(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a+1e-15) - (1 - y) * np.log(1 - a+1e-15)))

    @staticmethod
    def cross_entropy_derivative(z, a, y):
        return a - y

    @staticmethod
    def mse(a, y):
        return np.sum((a - y) ** 2)

    mse_derivative = cross_entropy_derivative

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
