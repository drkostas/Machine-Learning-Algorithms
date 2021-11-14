import matplotlib.pyplot as plt
import numpy as np
from typing import *


def plot_results(title: str, losses: List, test_accuracy: float, accuracies: List,
                 times: List, subsample: int = 1):
    losses_ = losses[::subsample]
    losses_ = list(zip(*losses_))
    loss_names = []
    loss_values = []

    for loss in losses_:
        loss_name, values = list(zip(*loss))
        loss_names.append(loss_name[0])
        loss_values.append(values)

    accuracies_ = accuracies[::subsample]
    times_ = times[::subsample]
    x = np.arange(1, len(accuracies_) + 1)

    fig, ax = plt.subplots(2, 2, figsize=(11, 4))
    sup_title = f"{title}\nMax Train Accuracy: {max(accuracies) * 100:.4f} %\n"
    if test_accuracy is not None:
        sup_title += f"\nTest Accuracy: {test_accuracy * 100:.4f} %"
    fig.suptitle(sup_title)
    # Accuracies
    ax[0][0].plot(x, accuracies_)
    ax[0][0].set_title(f'Accuracies per epoch')
    ax[0][0].set_xlabel("Epoch")
    ax[0][0].set_ylabel("Accuracies (%)")
    ax[0][0].set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax[0][0].set_yticklabels([0, 25., 50., 75., 100.])
    x_ticks = ax[0][0].get_xticks().tolist()
    ax[0][0].set_xticklabels([int(x_tick * subsample) for x_tick in x_ticks])
    ax[0][0].grid(True)
    # Times
    ax[0][1].plot(x, times_)
    ax[0][1].set_title(f'Times per epoch')
    ax[0][1].set_xlabel("Epoch")
    ax[0][1].set_ylabel("Second(s)")
    x_ticks = ax[0][1].get_xticks().tolist()
    ax[0][1].set_xticklabels([int(x_tick * subsample) for x_tick in x_ticks])
    ax[0][1].grid(True)
    for ind, (name, loss) in enumerate(zip(loss_names, loss_values)):
        ax[1][ind].plot(x, loss)
        ax[1][ind].set_title(f'{name} per epoch')
        ax[1][ind].set_xlabel("Epoch")
        ax[1][ind].set_ylabel(name)
        x_ticks = ax[1][ind].get_xticks().tolist()
        ax[1][ind].set_xticklabels([int(x_tick * subsample) for x_tick in x_ticks])
        ax[1][ind].grid(True)
    fig.tight_layout()
    make_space_above(ax, top_margin=1)


def make_space_above(axes, top_margin=1):
    """ Increase figure size to make top_margin (in inches) space for
        titles, without changing the axes sizes"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1 - s.top) * h + top_margin
    fig.subplots_adjust(bottom=s.bottom * h / figh, top=1 - top_margin / figh)
    fig.set_figheight(figh)
