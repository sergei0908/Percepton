"""Однослойный перцептрон с пороговой функцией активации."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


class Perceptron:
    """Бинарный линейный классификатор, обучаемый правилом перцептрона."""

    def __init__(
        self, n_features: int, learning_rate: float = 0.1, epochs: int = 50
    ) -> None:
        """Создать необученный перцептрон с нулевыми весами и смещением."""
        if n_features <= 0:
            raise ValueError("parameter n_features must be positive")
        if learning_rate <= 0:
            raise ValueError("parameter learning_rate be positive")
        if epochs <= 0:
            raise ValueError("parameter epochs must be positive")

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights: FloatArray = np.zeros(n_features)
        self.bias: float = 0.0
        self.errors_: list[int] = []  # число ошибок на каждой эпохе

    @staticmethod
    def activate(z: float) -> int:
        """Пороговая активация: вернуть 1, если z >= 0, иначе 0."""
        if z >= 0:
            return 1
        return 0

    def predict(self, x: FloatArray) -> int:
        """Предсказать класс (0 или 1) для одного объекта."""
        z = float(np.dot(self.weights, x)) + self.bias
        return self.activate(z)

    def train(self, X: FloatArray, y: IntArray) -> None:
        """Обучить веса по правилу перцептрона, запоминая число ошибок за эпоху."""
        self._check_shapes(X, y)
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y, strict=True):
                update = self.learning_rate * (int(target) - self.predict(xi))
                if update != 0:
                    self.weights += update * xi
                    self.bias += update
                    errors += 1
            self.errors_.append(errors)
            if errors == 0:
                break

    def score(self, X: FloatArray, y: IntArray) -> float:
        """Вернуть точность (accuracy): долю верно классифицированных объектов."""
        self._check_shapes(X, y)
        correct = 0
        for xi, target in zip(X, y, strict=True):
            pred = self.predict(xi)
            if pred == target:
                correct += 1
        return correct / len(y)

    def _check_shapes(self, X: FloatArray, y: IntArray) -> None:
        """Проверить, что X имеет форму (n_samples, n_features) и согласован с y."""
        if X.ndim != 2 and X.shape[1] != self.n_features:
            raise ValueError("X must be (n_samples, n_features)")
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        if len(y) == 0:
            raise ValueError("dataset is empty")
