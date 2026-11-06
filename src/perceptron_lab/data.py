"""Генерация синтетических данных для обучения модели."""

import numpy as np

from perceptron_lab.model import FloatArray, IntArray


def make_blobs(
    n_per_class: int = 50,
    centers: tuple[tuple[float, float], tuple[float, float]] = (
        (-2.0, -2.0),
        (2.0, 2.0),
    ),
    spread: float = 0.8,
    seed: int | None = None,
) -> tuple[FloatArray, IntArray]:
    """Сгенерировать два облака 2D-точек вокруг двух центров.

    Точки вокруг первого центра получают метку 0, вокруг второго — метку 1.
    """
    if n_per_class <= 0 or spread <= 0:
        msg = "n_per_class и spread должны быть положительными"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    cloud_0 = rng.normal(loc=centers[0], scale=spread, size=(n_per_class, 2))
    cloud_1 = rng.normal(loc=centers[1], scale=spread, size=(n_per_class, 2))
    X = np.vstack([cloud_0, cloud_1])
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int_)
    order = rng.permutation(len(X))
    return X[order], y[order]
