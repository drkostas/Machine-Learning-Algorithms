import matplotlib.pyplot as plt
import numpy as np


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
        ax[1].set_title("pX1 Histogram")
        ax[1].set_xlabel("pX1")
        ax[1].set_ylabel("Count")
        # ax[1].margins(0.1)  # 1% padding in all directions
        ax[1].legend({"Class 1": "tab:blue", "Class 2": "tab:orange"})
        ax[1].grid(True)
        # Fig config
        fig.tight_layout()