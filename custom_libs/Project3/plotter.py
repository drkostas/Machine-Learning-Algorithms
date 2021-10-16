import matplotlib.pyplot as plt
import numpy as np
from typing import *


class Plotter:
    nx: np.ndarray
    fx: np.ndarray
    px: np.ndarray
    px1: np.ndarray

    def __init__(self, nx: np.ndarray, fx: np.ndarray, px1: np.ndarray, px: np.ndarray):
        self.nx = nx
        self.fx = fx
        self.px = px
        self.px1 = px1

    def plot_fx_px1_histograms(self, bins=10):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        # fX histogram
        fx_c1 = self.fx[self.fx[:, -1] == 0][:, :-1].flatten()
        fx_c2 = self.fx[self.fx[:, -1] == 1][:, :-1].flatten()
        ax[0].hist([fx_c1, fx_c2], stacked=True, color=["tab:blue", "tab:orange"], bins=bins)
        ax[0].set_title("fX Histogram")
        ax[0].set_xlabel("fX")
        ax[0].set_ylabel("Count")
        # ax[0].margins(0.1)  # 1% padding in all directions
        ax[0].legend({"Class 1": "tab:blue", "Class 2": "tab:orange"})
        ax[0].grid(True)
        # fX histogram
        px1_c1 = self.px1[self.px1[:, -1] == 0][:, :-1].flatten()
        px1_c2 = self.px1[self.px1[:, -1] == 1][:, :-1].flatten()
        ax[1].hist([px1_c1, px1_c2], stacked=True, color=["tab:blue", "tab:orange"], bins=bins)
        # Annotate Plot
        ax[1].set_title("pX1 Histogram")
        ax[1].set_xlabel("pX1")
        ax[1].set_ylabel("Count")
        # ax[1].margins(0.1)  # 1% padding in all directions
        ax[1].legend({"Class 1": "tab:blue", "Class 2": "tab:orange"})
        ax[1].grid(True)
        # Fig config
        fig.tight_layout()

    @staticmethod
    def plot_roc(confusion_matrix_data: List[Dict]):
        cm_data_sorted = sorted(confusion_matrix_data, key=lambda row: row['fpr'])
        x = [cm_row['fpr'] for cm_row in cm_data_sorted]
        y = [cm_row['tpr'] for cm_row in cm_data_sorted]
        point_labels = [cm_row['priors'] for cm_row in cm_data_sorted]
        fig, ax = plt.subplots(1, 1, figsize=(11, 11))
        ax.plot(x, y, '-', color='tab:orange')
        previous_point = 0
        y_step = 0.3
        x_step = 0.01
        for px, py, pl in zip(x, y, point_labels):
            # Don't annotate everything
            if abs(previous_point - px) > 0.1:
                previous_point = px
                pxl = px - x_step
                pyl = py - y_step
                y_step -= 0.032
                x_step += 0.008
                pl_round = f'Priors: ({pl[0]:.2f}, {pl[1]:.2f})'
                ax.annotate(pl_round, xy=(px, py), xytext=(pxl, pyl),
                            bbox=dict(boxstyle="round", fc="none", ec="gray"),
                            arrowprops=dict(facecolor='black', arrowstyle="fancy",
                                            fc="0.6", ec="none",
                                            connectionstyle="angle3,angleA=0,angleB=-90"))
        # Annotate Plot
        ax.set_title('ROC for pX using Case 3 for different priors')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xbound(0, 1)
        ax.set_ybound(0, 1)
        ax.grid(True)
        ax.set_aspect('equal')
        # Fig Config
        fig.tight_layout()
