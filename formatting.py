"""Texto LaTeX de los modelos primal y dual y de la recuperación del juego."""

from __future__ import annotations

import numpy as np

V_PRIME = r"v^{\prime *}"
W_STAR = r"W^*"
Z_STAR = r"Z^*"
P_STAR = r"p^*"
Q_STAR = r"q^*"
V_STAR = r"v^*"


def display_number(value: float, significant: int = 6) -> str:
    """Número compacto para tablas y métricas."""

    number = float(value)
    if not np.isfinite(number):
        return "no disponible"
    if abs(number) <= 1e-12:
        return "0"
    nearest = round(number)
    if abs(number - nearest) <= 1e-8 * max(1.0, abs(number)):
        return str(int(nearest))
    return f"{number:.{significant}g}"


def latex_number(value: float, significant: int = 6) -> str:
    """Número seguro para KaTeX, incluida la notación científica."""

    number = float(value)
    if not np.isfinite(number):
        return r"\mathrm{NaN}"
    if abs(number) <= 1e-12:
        return "0"
    nearest = round(number)
    if abs(number - nearest) <= 1e-8 * max(1.0, abs(number)):
        return str(int(nearest))
    text = f"{number:.{significant}g}"
    if "e" not in text.lower():
        return text
    mantissa, exponent = f"{number:.{significant}e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa} \\times 10^{{{int(exponent)}}}"


def vector_tuple(values: np.ndarray) -> str:
    components = r",\ ".join(latex_number(component) for component in values)
    return "(" + components + ")"


def matrix_latex(symbol: str, values: np.ndarray) -> str:
    rows = [
        " & ".join(latex_number(entry) for entry in row)
        for row in np.asarray(values, dtype=float)
    ]
    body = r" \\ ".join(rows)
    return symbol + r" = \begin{pmatrix} " + body + r" \end{pmatrix}"


def translation_latex(minimum: float, k: float) -> str:
    if k == 0:
        return r"k = 0"
    return (
        r"k = 1 - \min_{i,j}\, a_{ij} = 1 - ("
        + latex_number(minimum)
        + r") = "
        + latex_number(k)
    )


def cell_shift_latex() -> str:
    return r"a^{\prime}_{ij} = a_{ij} + k"


def primal_substitution_latex() -> str:
    return r"x_i = \frac{p_i}{" + V_PRIME + "}"


def dual_substitution_latex() -> str:
    return r"y_j = \frac{q_j}{" + V_PRIME + "}"


def primal_symbolic_latex(rows: int, columns: int) -> str:
    return _aligned(
        [
            r"\text{Minimizar } & W = \sum_{i=1}^{" + str(rows) + r"} x_i",
            (
                r"\text{sujeto a } & \sum_{i=1}^{"
                + str(rows)
                + r"} a^{\prime}_{ij}\, x_i \ge 1, \quad j = 1,\ldots, "
                + str(columns)
            ),
            r"& x_i \ge 0, \quad i = 1,\ldots, " + str(rows),
        ]
    )


def dual_symbolic_latex(rows: int, columns: int) -> str:
    return _aligned(
        [
            r"\text{Maximizar } & Z = \sum_{j=1}^{" + str(columns) + r"} y_j",
            (
                r"\text{sujeto a } & \sum_{j=1}^{"
                + str(columns)
                + r"} a^{\prime}_{ij}\, y_j \le 1, \quad i = 1,\ldots, "
                + str(rows)
            ),
            r"& y_j \ge 0, \quad j = 1,\ldots, " + str(columns),
        ]
    )


def primal_numeric_latex(shifted: np.ndarray) -> str:
    row_count = shifted.shape[0]
    objective = " + ".join(f"x_{{{index}}}" for index in range(1, row_count + 1))
    lines = [r"\text{Minimizar } & W = " + objective]
    for column in range(shifted.shape[1]):
        prefix = r"\text{sujeto a } & " if column == 0 else "& "
        lines.append(
            prefix + linear_expression(shifted[:, column], "x") + r" \ge 1"
        )
    lines.append("& " + _nonnegative(row_count, "x"))
    return _aligned(lines)


def dual_numeric_latex(shifted: np.ndarray) -> str:
    column_count = shifted.shape[1]
    objective = " + ".join(f"y_{{{index}}}" for index in range(1, column_count + 1))
    lines = [r"\text{Maximizar } & Z = " + objective]
    for row in range(shifted.shape[0]):
        prefix = r"\text{sujeto a } & " if row == 0 else "& "
        lines.append(prefix + linear_expression(shifted[row, :], "y") + r" \le 1")
    lines.append("& " + _nonnegative(column_count, "y"))
    return _aligned(lines)


def optimum_latex(symbol: str, values: np.ndarray, objective: str, optimal: float) -> str:
    return (
        symbol
        + r"^* = "
        + vector_tuple(values)
        + r", \qquad "
        + objective
        + r"^* = "
        + latex_number(optimal)
    )


def value_recovery_latex(weight: float, value_prime: float, k: float, value: float) -> str:
    return _aligned(
        [
            V_PRIME
            + r" &= \frac{1}{"
            + W_STAR
            + r"} = \frac{1}{"
            + latex_number(weight)
            + r"} = "
            + latex_number(value_prime),
            V_STAR
            + r" &= "
            + V_PRIME
            + r" - k = "
            + latex_number(value_prime)
            + r" - "
            + latex_number(k)
            + r" = "
            + latex_number(value),
        ]
    )


def strategy_recovery_latex(
    x: np.ndarray,
    y: np.ndarray,
    value_prime: float,
    p: np.ndarray,
    q: np.ndarray,
) -> str:
    scale = latex_number(value_prime)
    return _aligned(
        [
            r"x_i &= \frac{p_i}{"
            + V_PRIME
            + r"} \quad \Rightarrow \quad p_i = x_i \cdot "
            + V_PRIME,
            P_STAR
            + r" &= "
            + scale
            + r" \cdot "
            + vector_tuple(x)
            + r" = "
            + vector_tuple(p),
            r"y_j &= \frac{q_j}{"
            + V_PRIME
            + r"} \quad \Rightarrow \quad q_j = y_j \cdot "
            + V_PRIME,
            Q_STAR
            + r" &= "
            + scale
            + r" \cdot "
            + vector_tuple(y)
            + r" = "
            + vector_tuple(q),
        ]
    )


def linear_expression(coefficients: np.ndarray, variable: str) -> str:
    """Suma lineal. Los coeficientes nulos se omiten; si todos lo son, queda 0."""

    parts: list[str] = []
    for index, coefficient in enumerate(np.asarray(coefficients, dtype=float), start=1):
        if coefficient == 0:
            continue
        magnitude = latex_number(abs(float(coefficient)))
        atom = f"{variable}_{{{index}}}"
        term = atom if magnitude == "1" else f"{magnitude} {atom}"
        if not parts:
            parts.append(f"-{term}" if coefficient < 0 else term)
        elif coefficient < 0:
            parts.append(f"- {term}")
        else:
            parts.append(f"+ {term}")
    if not parts:
        return "0"
    return " ".join(parts)


def _nonnegative(count: int, variable: str) -> str:
    return r",\ ".join(
        f"{variable}_{{{index}}} \\ge 0" for index in range(1, count + 1)
    )


def _aligned(lines: list[str]) -> str:
    return r"\begin{aligned} " + r" \\ ".join(lines) + r" \end{aligned}"
