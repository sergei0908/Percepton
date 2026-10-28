import numpy as np
import pytest

from perceptron_lab.model import Perceptron


def test_weights_have_expected_length() -> None:
    model = Perceptron(n_features=3)
    assert model.weights.shape == (3,)
    assert model.bias == 0.0


@pytest.mark.parametrize(("z", "expected"), [(0.0, 1), (0.5, 1), (-1.0, 0), (-1e-9, 0)])
def test_activate_threshold(z: float, expected: int) -> None:
    assert Perceptron.activate(z) == expected


def test_predict_with_fixed_weights() -> None:
    model = Perceptron(n_features=2)
    model.weights = np.array([1.0, 1.0])
    model.bias = -1.0
    assert model.predict(np.array([2.0, 2.0])) == 1
    assert model.predict(np.array([0.0, 0.0])) == 0


def test_train_separates_linearly_separable_data() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 0, 1])  # logical AND
    model = Perceptron(n_features=2, learning_rate=0.1, epochs=100)
    model.train(X, y)
    assert model.score(X, y) == 1.0


def test_train_records_errors_per_epoch() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 0, 1])
    model = Perceptron(n_features=2, epochs=100)
    model.train(X, y)
    assert len(model.errors_) >= 1
    assert model.errors_[-1] == 0


def test_score_is_fraction_of_correct() -> None:
    model = Perceptron(n_features=1)  # zero weights -> always predicts 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1, 1, 0, 0])
    assert model.score(X, y) == 0.5


def test_invalid_hyperparameters_raise() -> None:
    with pytest.raises(ValueError):
        Perceptron(n_features=0)
    with pytest.raises(ValueError):
        Perceptron(n_features=2, learning_rate=0)
    with pytest.raises(ValueError):
        Perceptron(n_features=2, epochs=0)


def test_shape_mismatch_raises() -> None:
    model = Perceptron(n_features=2)
    with pytest.raises(ValueError):
        model.train(np.array([[1.0, 2.0, 3.0]]), np.array([1]))
