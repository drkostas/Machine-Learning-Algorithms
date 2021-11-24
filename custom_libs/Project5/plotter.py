import matplotlib.pyplot as plt
import numpy as np
from typing import *


def plot_task3_errors(human: List, default: List, pid: List, qlearning: List):
    fig, ax = plt.subplots(4, 1, figsize=(11, 14))
    plot_color = 'dodgerblue'
    # Human control
    ax[0].plot(np.arange(len(human)), human, color=plot_color)
    ax[0].margins(0.1)  # 1% padding in all directions
    ax[0].set_title("Human Control Error vs Ticks")
    ax[0].set_xlabel("Number of ticks")
    ax[0].set_ylabel("Error")
    ax[0].set_xticks(np.arange(0, 600, 50))
    ax[0].grid(False)
    # Human control
    ax[1].plot(np.arange(len(default)), default, color=plot_color)
    ax[1].margins(0.1)  # 1% padding in all directions
    ax[1].set_title("Demo/Default Error vs Ticks")
    ax[1].set_xlabel("Number of ticks")
    ax[1].set_ylabel("Error")
    ax[1].set_xticks(np.arange(0, 600, 50))
    ax[1].grid(False)
    # Human control
    ax[2].plot(np.arange(len(pid)), pid, color=plot_color)
    ax[2].margins(0.1)  # 1% padding in all directions
    ax[2].set_title("PID Error vs Ticks")
    ax[2].set_xlabel("Number of ticks")
    ax[2].set_ylabel("Error")
    ax[2].set_xticks(np.arange(0, 600, 50))
    ax[2].grid(False)
    # Human control
    ax[3].plot(np.arange(len(qlearning)), qlearning, color=plot_color)
    ax[3].margins(0.1)  # 1% padding in all directions
    ax[3].set_title("Qlearning Error vs Ticks")
    ax[3].set_xlabel("Number of ticks")
    ax[3].set_ylabel("Error")
    ax[3].set_xticks(np.arange(0, 600, 50))
    ax[3].grid(False)
    # Fig Config
    fig.tight_layout()

def plot_bonus_task2_errors(errors):
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    plot_color = 'dodgerblue'
    # Human control
    ax.plot(np.arange(len(errors)), errors, color=plot_color)
    ax.margins(0.1)  # 1% padding in all directions
    ax.set_title("Linear Regression Error vs Ticks")
    ax.set_xlabel("Number of ticks")
    ax.set_ylabel("Error")
    ax.set_xticks(np.arange(0, 600, 50))
    ax.grid(False)