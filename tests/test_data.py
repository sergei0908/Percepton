import pytest

from perceptron_lab.data import load_iris, load_iris_binary, make_blobs
from perceptron_lab.model import Perceptron


def test_blobs_shapes_and_labels() -> None:
    X, y = make_blobs(n_per_class=30, seed=0)
    assert X.shape == (60, 2)
    assert y.shape == (60,)
    assert set(y.tolist()) == {0, 1}
    assert (y == 0).sum() == 30


def test_blobs_invalid_args_raise() -> None:
    with pytest.raises(ValueError):
        make_blobs(n_per_class=0)
    with pytest.raises(ValueError):
        make_blobs(spread=0)


def test_perceptron_learns_blobs() -> None:
    X, y = make_blobs(seed=0)
    model = Perceptron(n_features=2, epochs=100)
    model.train(X, y)
    assert model.score(X, y) == 1.0


def test_load_iris_shape() -> None:
    X, species = load_iris()
    assert X.shape == (150, 4)
    assert len(species) == 150
    assert set(species) == {"setosa", "versicolor", "virginica"}


def test_iris_one_vs_rest() -> None:
    X, y = load_iris_binary("setosa")
    assert X.shape == (150, 4)
    assert y.sum() == 50  # 50 setosa -> label 1, other 100 -> label 0


def test_iris_two_classes() -> None:
    X, y = load_iris_binary("versicolor", "virginica")
    assert X.shape == (100, 4)
    assert y.sum() == 50


def test_iris_bad_class_raises() -> None:
    with pytest.raises(ValueError):
        load_iris_binary("rose")
    with pytest.raises(ValueError):
        load_iris_binary("setosa", "setosa")


def test_perceptron_separates_setosa() -> None:
    X, y = load_iris_binary("setosa")
    model = Perceptron(n_features=4, epochs=100)
    model.train(X, y)
    assert model.score(X, y) == 1.0
