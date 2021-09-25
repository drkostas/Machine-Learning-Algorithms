import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


class Plotter:
    synth_tr: np.ndarray
    synth_te: np.ndarray
    pima_tr: np.ndarray
    pima_te: np.ndarray

    def __init__(self, synth_tr: np.ndarray, synth_te: np.ndarray, pima_tr: np.ndarray,
                 pima_te: np.ndarray):
        self.synth_tr = synth_tr
        self.synth_te = synth_te
        self.pima_tr = pima_tr
        self.pima_te = pima_te

    def plot_dataset(self):
        fig, ax = plt.subplots(1, 3, figsize=(11, 4))
        plot_color = 'dodgerblue'
        # synth_tr f1-f2 Scatter Plot
        ax[0].scatter(self.synth_tr[:, 0][self.synth_tr[:, -1] == 0],
                      self.synth_tr[:, 1][self.synth_tr[:, -1] == 0],
                      color='royalblue', s=12, marker='o', label="Class 0")
        ax[0].scatter(self.synth_tr[:, 0][self.synth_tr[:, -1] == 1],
                      self.synth_tr[:, 1][self.synth_tr[:, -1] == 1],
                      color='red', s=12, marker='o', label="Class 1")
        ax[0].margins(0.1)  # 1% padding in all directions
        ax[0].set_title("Synth Dataset Scatter Plot")
        ax[0].set_xlabel("Feature 1")
        ax[0].set_ylabel("Feature 2")
        ax[0].legend()
        ax[0].grid(True)
        # f1 Hist
        hist, bins, patches = ax[1].hist(self.synth_tr[:, 0], density=True, bins=20, color=plot_color,
                                         edgecolor='black',
                                         linewidth=0.5)  # density=False would make counts
        ax[1].set_title("Synth Dataset Density Histogram")
        ax[1].set_xlabel("Feature 1")
        ax[1].set_ylabel("Density")
        ax[1].margins(0.1)  # 1% padding in all directions
        # f2 Hist
        hist, bins, patches = ax[2].hist(self.synth_tr[:, 1], density=True, bins=20, color=plot_color,
                                         edgecolor='black',
                                         linewidth=0.5)  # density=False would make counts
        ax[2].set_title("Synth Dataset Density Histogram")
        ax[2].set_xlabel("Feature 2")
        ax[2].set_ylabel("Density")
        ax[2].margins(0.1)  # 1% padding in all directions
        fig.tight_layout()
        fig.show()

    @staticmethod
    def plot_knn_overall_accuracies(synth_k_range, synth_accuracies, pima_k_range, pima_accuracies):
        fig, ax = plt.subplots(2, 1, figsize=(9, 9))

        # Synth Dataset
        ax[0].plot(synth_k_range, synth_accuracies, label='Synthetic Dataset', color='deepskyblue')
        ax[0].set_title('Overall Classification accuracy vs k for the Synthetic Dataset')
        ax[0].set_xlabel('k')
        ax[0].set_ylabel('Overall Classification Accuracy')
        _ = ax[0].set_xticks(synth_k_range)
        ax[0].legend()
        # Pima Dataset
        ax[1].plot(pima_k_range, pima_accuracies, label='Pima Dataset', color='orange')
        ax[1].set_title('Overall Classification accuracy vs k for the Pima Dataset')
        ax[1].set_xlabel('k')
        ax[1].set_ylabel('Overall Classification Accuracy')
        _ = ax[1].set_xticks(pima_k_range)
        ax[1].legend()
        # Show plot
        fig.tight_layout()
        fig.show()

    @staticmethod
    def plot_decision_boundaries(knn, h: float = 0.2):
        # Init values statically from Project 1
        a_eucl = -0.8326229483927666
        b_eucl = 0.44378197841356054
        a_maha = -0.13486408662390306
        b_maha = 0.49454949088419903
        A = -2.9353736949690252
        B = -7.122064910873636
        C = -9.131232270572491
        D = -4.023021305932989
        E = 29.777685196099192
        F = -14.251862334038359
        means = np.array([[-0.22147024, 0.32575494], [0.07595431, 0.68296891]])
        means_center = np.array([-0.07275796159999995, 0.5043619269200001])
        a_m = 1.2010238270880302
        b_m = 0.591745972411956
        # Plot the Decision Boundaries
        fig, ax = plt.subplots(1, 1, figsize=(11, 9))
        eucl_x_range = np.linspace(-0.8, 0.9, 50)
        maha_x_range = np.linspace(-1, 1, 50)
        quadr_x_range = np.linspace(-1.1, 1.1, 50)
        quadr_y_range = np.linspace(-0.2, 1.1, 50)
        # KNN Decision Boundaries
        cmap_light = ListedColormap(['lightblue', 'moccasin'])
        # KNN Decision Boundaries
        x, y = knn.train_x, knn.train_y
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        x_target = np.c_[xx.ravel(), yy.ravel()]
        Z = knn.predict(x_target, only_x=True)
        Z = Z.reshape(xx.shape)
        knn_contour_plot = ax.contourf(xx, yy, Z, cmap=cmap_light)
        # Class 0 Scatter plot
        ax.scatter(x[:, 0][y == 0], x[:, 1][y == 0],
                   color='royalblue', s=10, label='Class 0')
        # Class 1 Scatter plot
        ax.scatter(x[:, 0][y == 1], x[:, 1][y == 1],
                   color='red', s=10, label='Class 1')
        # Decision Boundaries
        # Euclidean
        ax.plot(eucl_x_range, a_eucl * eucl_x_range + b_eucl, color='orange',
                label=f'Euclidean Decision Boundary')
        # Mahalanobis
        ax.plot(maha_x_range, a_maha * maha_x_range + b_maha, color='deepskyblue',
                label=f'Mahalanobis Decision Boundary')
        # Quadratic
        x_quad, y_quad = np.meshgrid(quadr_x_range, quadr_y_range)
        quadr_equation = A * x_quad ** 2 + B * y_quad ** 2 + C * x_quad * y_quad + D * x_quad + E * y_quad + F
        quad_contour_plt = ax.contour(x_quad, y_quad, quadr_equation, [0],
                                 colors='limegreen')
        ax.clabel(quad_contour_plt, inline=1, fontsize=10)
        quad_contour_plt.collections[0].set_label('Quadratic Decision Boundary')
        # Line that links the means of the two classes
        mline_x_range = np.linspace(means[0][0], means[1][0], 5)
        ax.plot(mline_x_range, a_m * mline_x_range + b_m,
                color='m', linestyle='dashed', label='Line linking the two means')
        # Class 0 Mean value
        ax.plot(means[0][0], means[0][1],
                'bo', markersize=11, markeredgecolor='w', label='Class 0 Mean value')
        # Class 1 Mean value
        ax.plot(means[1][0], means[1][1],
                'ro', markersize=11, markeredgecolor='w', label='Class 1 Mean value')
        # Center of the linking line
        ax.plot(means_center[0], means_center[1],
                'mo', markersize=11, markeredgecolor='w',
                label=f'Center of the linking line')
        # Show figure
        ax.set_title(
            "The three Decision Boundaries plotted against the scatter plot of the two features")
        # ax.axis('equal')
        ax.set_xlim(-1.35, 1.3)
        ax.set_ylim(-0.35, 1.15)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend(loc='upper left')
        # ax.margins(0.1)
        fig.show()

    @staticmethod
    def plot_membership_changes(kmeans_membership_changes, wta_membership_changes, epsilon):
        fig, ax = plt.subplots(2, 1, figsize=(9, 9))

        # Pima, Kmeans
        kmeans_range = range(2, len(kmeans_membership_changes)+2)
        ax[0].plot(kmeans_range, kmeans_membership_changes,
                   label=f'Kmeans', color='deepskyblue')
        ax[0].set_title('Membership Changes per epoch for Kmeans on Pima Dataset')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Membership Changes')
        _ = ax[0].set_xticks(kmeans_range)
        ax[0].legend()
        # Pima, WTA
        wta_range = range(2, len(wta_membership_changes) + 2)
        ax[1].plot(wta_range, wta_membership_changes,
                   label=f'WTA: epsilon={epsilon}', color='orange')
        ax[1].set_title('Membership Changes per epoch for WTA on Pima Dataset')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Membership Changes')
        _ = ax[1].set_xticks(wta_range)
        ax[1].legend()
        # Show plot
        fig.tight_layout()
        fig.show()