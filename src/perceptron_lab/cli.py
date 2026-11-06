"""Командная строка: сгенерировать данные, обучить перцептрон, вывести точность."""

import argparse
from pathlib import Path

from perceptron_lab.data import make_blobs
from perceptron_lab.model import Perceptron
from perceptron_lab.plots import plot_decision_boundary, plot_errors


def positive_int(text: str) -> int:
    """Преобразовать строку в целое число больше нуля."""
    try:
        value = int(text)
    except ValueError:
        msg = f"ожидалось целое число, получено {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if value <= 0:
        msg = f"значение должно быть больше 0, получено {value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def positive_float(text: str) -> float:
    """Преобразовать строку в дробное число больше нуля."""
    try:
        value = float(text)
    except ValueError:
        msg = f"ожидалось число, получено {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if value <= 0:
        msg = f"значение должно быть больше 0, получено {value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def build_parser() -> argparse.ArgumentParser:
    """Описать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Обучить перцептрон на двух облаках точек."
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=200,
        help="всего точек (по умолчанию 200)",
    )
    parser.add_argument(
        "--lr", type=positive_float, default=0.1, help="скорость обучения (0.1)"
    )
    parser.add_argument(
        "--epochs", type=positive_int, default=50, help="максимум эпох (50)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="зерно случайности для повторяемости"
    )
    parser.add_argument(
        "--plots", type=Path, default=None, help="папка, куда сохранить графики"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Запустить программу; вернуть код выхода."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.samples < 2:
        parser.error("--samples должно быть не меньше 2 (по точке на класс)")

    X, y = make_blobs(n_per_class=args.samples // 2, seed=args.seed)
    model = Perceptron(n_features=2, learning_rate=args.lr, epochs=args.epochs)
    model.train(X, y)

    print(f"Точек: {len(X)}, эпох пройдено: {len(model.errors_)}")
    print(f"Точность: {model.score(X, y):.3f}")

    if args.plots is not None:
        args.plots.mkdir(parents=True, exist_ok=True)
        plot_errors(model.errors_, args.plots / "errors.png")
        plot_decision_boundary(model, X, y, args.plots / "boundary.png")
        print(f"Графики сохранены в {args.plots}")
    return 0
