"""Данные для перцептрона: синтетические облака точек и набор iris."""

import csv
import io
from importlib.resources import files

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


IRIS_CLASSES = ("setosa", "versicolor", "virginica")


def load_iris() -> tuple[FloatArray, list[str]]:
    """Прочитать iris.csv из пакета: 150 цветков, 4 признака и название вида."""
    text = files("perceptron_lab").joinpath("iris.csv").read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    X = np.array(
        [
            [float(value) for key, value in row.items() if key != "species"]
            for row in rows
        ],
        dtype=np.float64,
    )
    species = [row["species"] for row in rows]
    return X, species


def load_iris_binary(
    positive: str, negative: str | None = None
) -> tuple[FloatArray, IntArray]:
    """Сделать из iris задачу с двумя классами.

    Вид positive получает метку 1. Если negative не задан, все остальные
    виды получают метку 0 (один против всех). Если задан — берутся только
    два вида: positive (1) и negative (0).

    Raises:
        ValueError: если вид неизвестен или positive совпадает с negative.

    """
    for name in (positive, negative):
        if name is not None and name not in IRIS_CLASSES:
            msg = f"неизвестный вид {name!r}, допустимо: {', '.join(IRIS_CLASSES)}"
            raise ValueError(msg)
    if positive == negative:
        msg = "positive и negative должны быть разными видами"
        raise ValueError(msg)

    X, species = load_iris()
    labels = np.array(species)
    if negative is not None:
        mask = (labels == positive) | (labels == negative)
        X, labels = X[mask], labels[mask]
    y = (labels == positive).astype(np.int_)
    return X, y
