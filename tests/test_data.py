import pytest

from perceptron_lab.data import make_blobs
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
