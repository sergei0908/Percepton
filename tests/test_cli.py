from pathlib import Path

import pytest

from perceptron_lab.cli import main


def test_cli_prints_accuracy(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--samples", "100", "--lr", "0.1", "--epochs", "50", "--seed", "0"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Точность: 1.000" in out


def test_cli_saves_plots(tmp_path: Path) -> None:
    main(["--seed", "0", "--plots", str(tmp_path)])
    assert (tmp_path / "errors.png").exists()
    assert (tmp_path / "boundary.png").exists()


def test_cli_overlapping_clouds_not_perfect(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--spread", "2.0", "--seed", "0"])
    out = capsys.readouterr().out
    assert "Точность: 1.000" not in out


def test_cli_iris_setosa(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--iris", "setosa", "--epochs", "100"])
    out = capsys.readouterr().out
    assert "Точек: 150" in out
    assert "Точность: 1.000" in out


def test_cli_iris_plots_only_errors(tmp_path: Path) -> None:
    main(["--iris", "versicolor", "virginica", "--plots", str(tmp_path)])
    assert (tmp_path / "errors.png").exists()
    assert not (tmp_path / "boundary.png").exists()  # 4 features: no 2D boundary


@pytest.mark.parametrize(
    "argv",
    [
        ["--samples", "abc"],
        ["--samples", "-5"],
        ["--samples", "1"],
        ["--lr", "0"],
        ["--epochs", "2.5"],
        ["--spread", "0"],
        ["--iris", "rose"],
        ["--iris", "setosa", "setosa"],
        ["--iris", "setosa", "versicolor", "virginica"],
    ],
)
def test_cli_bad_arguments_exit_with_message(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2
    assert "error" in capsys.readouterr().err
