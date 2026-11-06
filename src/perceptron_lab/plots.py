"""Графики: ошибки по эпохам и граница решения перцептрона."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # рисуем в файл, без окна (нужно для CI)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from perceptron_lab.model import FloatArray, IntArray, Perceptron


def plot_errors(errors: list[int], path: Path) -> None:
    """Сохранить в PNG график числа ошибок на каждой эпохе."""
    epochs = range(1, len(errors) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, errors, marker="o")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Число ошибок")
    ax.set_title("Ошибки перцептрона по эпохам")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(visible=True)
    fig.savefig(path)
    plt.close(fig)


def plot_decision_boundary(
    model: Perceptron, X: FloatArray, y: IntArray, path: Path
) -> None:
    """Сохранить в PNG точки двух классов и прямую w0·x + w1·y + b = 0."""
    w0, w1 = model.weights
    b = model.bias
    fig, ax = plt.subplots()
    ax.scatter(X[y == 0, 0], X[y == 0, 1], label="класс 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label="класс 1")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    if w1 != 0:
        xs = np.linspace(x_min, x_max, 100)
        ys = -(w0 * xs + b) / w1
        ax.plot(xs, ys, color="black", label="граница")
    elif w0 != 0:
        ax.axvline(-b / w0, color="black", label="граница")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Граница решения перцептрона")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
