"""Командная строка: подготовить данные, обучить перцептрон, вывести точность."""

import argparse
from pathlib import Path

from perceptron_lab.data import IRIS_CLASSES, load_iris_binary, make_blobs
from perceptron_lab.model import FloatArray, IntArray, Perceptron
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
        description="Обучить перцептрон на облаках точек или на iris."
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
        "--spread",
        type=positive_float,
        default=0.8,
        help="разброс точек в облаке (0.8); больше 1 — облака перекрываются",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="зерно случайности для повторяемости"
    )
    parser.add_argument(
        "--iris",
        nargs="+",
        choices=IRIS_CLASSES,
        metavar="SPECIES",
        default=None,
        help="обучить на iris: один вид (против остальных) или два вида",
    )
    parser.add_argument(
        "--plots", type=Path, default=None, help="папка, куда сохранить графики"
    )
    return parser


def load_data(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[FloatArray, IntArray]:
    """Вернуть (X, y): облака точек или iris, в зависимости от аргументов."""
    if args.iris is None:
        if args.samples < 2:
            parser.error("--samples должно быть не меньше 2 (по точке на класс)")
        return make_blobs(
            n_per_class=args.samples // 2, spread=args.spread, seed=args.seed
        )
    if len(args.iris) > 2:
        parser.error("--iris принимает один или два вида")
    if len(args.iris) == 2 and args.iris[0] == args.iris[1]:
        parser.error("--iris: два вида должны быть разными")
    negative = args.iris[1] if len(args.iris) == 2 else None
    return load_iris_binary(args.iris[0], negative)


def save_plots(model: Perceptron, X: FloatArray, y: IntArray, folder: Path) -> None:
    """Сохранить графики; границу решения — только для 2 признаков."""
    folder.mkdir(parents=True, exist_ok=True)
    plot_errors(model.errors_, folder / "errors.png")
    if model.n_features == 2:
        plot_decision_boundary(model, X, y, folder / "boundary.png")
    print(f"Графики сохранены в {folder}")


def main(argv: list[str] | None = None) -> int:
    """Запустить программу; вернуть код выхода (0 — успех)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    X, y = load_data(args, parser)

    model = Perceptron(n_features=X.shape[1], learning_rate=args.lr, epochs=args.epochs)
    model.train(X, y)
    print(f"Точек: {len(X)}, эпох пройдено: {len(model.errors_)}")
    print(f"Точность: {model.score(X, y):.3f}")

    if args.plots is not None:
        save_plots(model, X, y, args.plots)
    return 0
