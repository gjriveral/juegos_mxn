"""Programación lineal para juegos de suma cero en estrategias mixtas.

El Jugador 1 elige la fila de A y el Jugador 2 la columna. La entrada a_ij es el
pago que recibe el Jugador 1. Si algún pago no es estrictamente positivo, se
suma una constante k para obtener A' con a'_ij > 0. El valor del juego se
recupera al final restando esa misma constante.

Con v' > 0, el cambio de variable x_i = p_i / v' convierte el problema del
Jugador 1 en

    minimizar    W = sum_i x_i
    sujeto a     sum_i a'_ij x_i >= 1    para cada columna j
                 x_i >= 0

El valor trasladado es v' = 1 / W*. Las variables del Jugador 2 no se obtienen
resolviendo otro modelo: y_j* es el precio sombra de la restricción de la
columna j, y Z* = sum_j y_j* coincide con W* por dualidad fuerte.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

_NEGATIVE_TOLERANCE = 1e-7
_SNAP_RELATIVE = 1e-8
_OBJECTIVE_FLOOR = 1e-14
_DUALITY_TOLERANCE = 1e-5
_PROBABILITY_TOLERANCE = 1e-5
_PROBABILITY_SNAP = 1e-10


class GameSolverError(Exception):
    """El juego no pudo resolverse con el modelo lineal."""


class InvalidPayoffError(GameSolverError):
    """La matriz de pagos no es numérica, finita o estrictamente positiva tras la traslación."""


class InfeasibleProblemError(GameSolverError):
    """Ningún x >= 0 cumple las restricciones de columna."""


class UnboundedProblemError(GameSolverError):
    """El objetivo puede disminuir sin límite y no hay óptimo finito."""


@dataclass(frozen=True)
class TranslatedGame:
    """Matriz original y la traslación que hace estrictamente positivos sus pagos."""

    A: np.ndarray
    minimum: float
    k: float
    A_prime: np.ndarray


@dataclass(frozen=True)
class GameSolution:
    """Óptimo primal, precios sombra y estrategias mixtas del juego original."""

    translated: TranslatedGame
    x: np.ndarray
    W: float
    y: np.ndarray
    Z: float
    v_prime: float
    v: float
    p: np.ndarray
    q: np.ndarray
    method: str
    warnings: tuple[str, ...]


def translate_payoff(matrix: np.ndarray) -> TranslatedGame:
    """Elige k de modo que cada entrada de A' = A + k sea estrictamente positiva.

    Si el pago mínimo ya es positivo, k = 0. Si es negativo o cero,
    k = 1 - min(A): el pago más pequeño queda en 1 y, cuando A es entera, A'
    también lo es.
    """

    payoff = _as_payoff(matrix)
    minimum = float(np.min(payoff))
    if minimum <= 0:
        k = 1.0 - minimum
    else:
        k = 0.0
    if not np.isfinite(k):
        raise InvalidPayoffError(
            "Los pagos tienen una magnitud demasiado grande para trasladar la matriz."
        )
    shifted = payoff + k
    if not np.all(np.isfinite(shifted)) or not np.all(shifted > 0):
        raise InvalidPayoffError(
            "La traslación no produjo una matriz estrictamente positiva. "
            "Revisa la escala de los pagos."
        )
    return TranslatedGame(
        A=payoff,
        minimum=minimum,
        k=float(k),
        A_prime=shifted,
    )


def solve_translated_game(game: TranslatedGame) -> GameSolution:
    """Resuelve el primal y lee y* en los precios sombra de sus restricciones."""

    if not np.all(game.A_prime > 0):
        raise InvalidPayoffError(
            "La matriz trasladada debe tener todos sus pagos estrictamente positivos."
        )

    result, method = _optimize_primal(game.A_prime)
    row_count, column_count = game.A_prime.shape
    if result.x is None or result.ineqlin is None or result.ineqlin.marginals is None:
        raise GameSolverError(
            "El solver terminó sin devolver la solución primal o sus precios sombra."
        )

    raw_x = np.asarray(result.x, dtype=float)
    raw_marginals = np.asarray(result.ineqlin.marginals, dtype=float)
    if raw_x.shape != (row_count,):
        raise GameSolverError(
            f"Se esperaban {row_count} variables primales x_i y el solver devolvió {raw_x.size}."
        )
    if raw_marginals.shape != (column_count,):
        raise GameSolverError(
            "Los precios sombra deben traer exactamente "
            f"{column_count} componentes, una por cada estrategia del Jugador 2."
        )

    weight = float(result.fun)
    if not np.isfinite(weight) or weight <= _OBJECTIVE_FLOOR:
        raise GameSolverError(
            "W* no es un número positivo utilizable, así que no se puede calcular "
            "v' = 1/W*. Revisa la escala de la matriz de pagos."
        )

    # linprog recibe la restricción como -(A')^T x <= -1. El marginal de esa
    # forma <= es el opuesto del precio sombra de (A')^T x >= 1, y ese precio
    # sombra es exactamente la variable dual y_j.
    raw_y = -raw_marginals
    x = _snap_nonnegative(raw_x, weight)
    y = _snap_nonnegative(raw_y, max(weight, float(np.max(np.abs(raw_y)))))
    total_dual = float(np.sum(y))
    value_prime = 1.0 / weight
    value = value_prime - game.k
    row_strategy = _snap_probabilities(x * value_prime)
    column_strategy = _snap_probabilities(y * value_prime)
    if row_strategy.shape != (row_count,) or column_strategy.shape != (column_count,):
        raise GameSolverError(
            "Las probabilidades no conservan las dimensiones del juego: "
            f"p* debe tener longitud {row_count} y q* longitud {column_count}."
        )

    warnings: list[str] = []
    if abs(total_dual - weight) > _DUALITY_TOLERANCE * max(1.0, abs(weight)):
        warnings.append(
            "La suma de los precios sombra Z* no coincide con W* dentro de la tolerancia numérica."
        )
    if (
        abs(float(row_strategy.sum()) - 1.0) > _PROBABILITY_TOLERANCE
        or abs(float(column_strategy.sum()) - 1.0) > _PROBABILITY_TOLERANCE
    ):
        warnings.append(
            "Las probabilidades recuperadas no suman 1 dentro de la tolerancia numérica."
        )

    return GameSolution(
        translated=game,
        x=x,
        W=weight,
        y=y,
        Z=total_dual,
        v_prime=value_prime,
        v=value,
        p=row_strategy,
        q=column_strategy,
        method=method,
        warnings=tuple(warnings),
    )


def solve_zero_sum_game(matrix: np.ndarray) -> GameSolution:
    """Trasladar la matriz, resolver el primal y recuperar p*, q* y v*."""

    return solve_translated_game(translate_payoff(matrix))


def _as_payoff(matrix: np.ndarray) -> np.ndarray:
    try:
        payoff = np.array(matrix, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise InvalidPayoffError(
            "La matriz de pagos debe contener solo números."
        ) from error
    if payoff.ndim != 2 or payoff.shape[0] == 0 or payoff.shape[1] == 0:
        raise InvalidPayoffError(
            "Define al menos una estrategia para cada jugador."
        )
    if not np.all(np.isfinite(payoff)):
        raise InvalidPayoffError(
            "La matriz de pagos debe estar completa y contener solo números finitos."
        )
    return payoff


def _snap_nonnegative(vector: np.ndarray, scale: float) -> np.ndarray:
    cleaned = np.array(vector, dtype=float, copy=True)
    magnitude = max(1.0, abs(scale))
    if cleaned.size:
        magnitude = max(magnitude, float(np.max(np.abs(cleaned))))
    if np.any(cleaned < -_NEGATIVE_TOLERANCE * magnitude):
        raise GameSolverError(
            "La solución del solver tiene componentes negativas fuera de la tolerancia numérica."
        )
    cleaned[cleaned < 0.0] = 0.0
    cleaned[cleaned <= _SNAP_RELATIVE * magnitude] = 0.0
    return cleaned


def _snap_probabilities(probabilities: np.ndarray) -> np.ndarray:
    cleaned = np.array(probabilities, dtype=float, copy=True)
    cleaned[np.abs(cleaned) <= _PROBABILITY_SNAP] = 0.0
    cleaned[cleaned < 0.0] = 0.0
    return cleaned


def build_primal_program(
    shifted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, None]]]:
    """Arma el primal de un juego m × n ya trasladado.

    c tiene m unos. A_ub = -(A')^T tiene una fila por cada una de las n
    columnas del juego, y b_ub = -1 convierte sum_i a'_ij x_i >= 1 al formato
    de minimización de SciPy. Hay una cota (0, None) por cada variable x_i.
    """

    row_count, column_count = shifted.shape
    objective = np.ones(row_count)
    inequality = -shifted.T
    rhs = -np.ones(column_count)
    bounds = [(0, None) for _ in range(row_count)]
    return objective, inequality, rhs, bounds


def _optimize_primal(shifted: np.ndarray):
    """Minimiza c'x sujeto a A_ub x <= b_ub y x >= 0.

    Se intenta primero el simplex dual de HiGHS. Si el reporte es numérico o
    se agotan las iteraciones, se reintenta con la selección automática de HiGHS.
    Un primal infactible o no acotado se reporta de inmediato.
    """

    objective, inequality, rhs, bounds = build_primal_program(shifted)
    last_result = None
    for method in ("highs-ds", "highs"):
        try:
            last_result = linprog(
                objective,
                A_ub=inequality,
                b_ub=rhs,
                bounds=bounds,
                method=method,
            )
        except ValueError as error:
            raise GameSolverError(
                f"No se pudo construir el problema lineal ({method}): {error}"
            ) from error
        if last_result.success:
            return last_result, method
        if last_result.status in (2, 3):
            break
    if last_result is None:
        raise GameSolverError("No se pudo ejecutar el solver lineal.")
    _raise_solver_status(last_result)


def _raise_solver_status(result) -> None:
    detail = f" Detalle del solver: {result.message}"
    if result.status == 2:
        raise InfeasibleProblemError(
            "El problema lineal es infactible: no hay un vector x ≥ 0 que cumpla "
            "todas las restricciones de columna. No existe un valor óptimo W* "
            "y el juego no se puede recuperar con esta formulación."
            + detail
        )
    if result.status == 3:
        raise UnboundedProblemError(
            "El problema lineal no está acotado: W puede disminuir sin límite "
            "dentro de la región factible. No existe un óptimo finito y el juego "
            "no se puede recuperar con esta formulación."
            + detail
        )
    raise GameSolverError(
        "No se obtuvo una solución óptima del problema primal. "
        "Revisa la escala y los valores de la matriz de pagos."
        + detail
    )


def _assert_dimensions(payoff: np.ndarray, solution: GameSolution) -> None:
    row_count, column_count = payoff.shape
    objective, inequality, rhs, bounds = build_primal_program(solution.translated.A_prime)
    if objective.shape != (row_count,):
        raise AssertionError("c debe tener una componente por estrategia del Jugador 1.")
    if inequality.shape != (column_count, row_count):
        raise AssertionError("A_ub debe ser -(A')^T, de tamaño n × m.")
    if rhs.shape != (column_count,):
        raise AssertionError("b_ub debe tener una cota por columna.")
    if bounds != [(0, None) for _ in range(row_count)]:
        raise AssertionError("Las cotas deben ser m tuplas (0, None).")
    if solution.x.shape != (row_count,) or solution.p.shape != (row_count,):
        raise AssertionError("x* y p* deben tener longitud m.")
    if solution.y.shape != (column_count,) or solution.q.shape != (column_count,):
        raise AssertionError("y* y q* deben tener longitud n.")


def _assert_consistent(solution: GameSolution) -> None:
    shifted = solution.translated.A_prime
    coverage = shifted.T @ solution.x
    dual_rows = shifted @ solution.y
    if np.any(coverage < 1.0 - 1e-5):
        raise AssertionError("La solución primal no cubre las restricciones de columna.")
    if np.any(dual_rows > 1.0 + 1e-5):
        raise AssertionError("Los precios sombra no son factibles para el dual.")
    if abs(solution.W - solution.Z) > 1e-5:
        raise AssertionError("Se perdió la dualidad fuerte entre W* y Z*.")
    if abs(solution.v_prime - 1.0 / solution.W) > 1e-8:
        raise AssertionError("v' no es el recíproco de W*.")
    if abs(solution.v - (solution.v_prime - solution.translated.k)) > 1e-8:
        raise AssertionError("No se recuperó v* restando k.")
    if abs(float(solution.p.sum()) - 1.0) > 1e-5:
        raise AssertionError("p* no es una distribución.")
    if abs(float(solution.q.sum()) - 1.0) > 1e-5:
        raise AssertionError("q* no es una distribución.")
    expected = float(solution.p @ solution.translated.A @ solution.q)
    if abs(expected - solution.v) > 1e-5:
        raise AssertionError("p*^T A q* no reproduce el valor del juego.")
    expected_shifted = float(solution.p @ shifted @ solution.q)
    if abs(expected_shifted - solution.v_prime) > 1e-5:
        raise AssertionError("p*^T A' q* no reproduce el valor trasladado.")
    if solution.warnings:
        raise AssertionError(solution.warnings)


def run_self_check() -> None:
    """Comprueba juegos con valor conocido, traslación y los dos estados de error."""

    cases = (
        (
            [[3.0, -1.0], [-2.0, 4.0]],
            dict(k=3.0, value=1.0, p=[0.6, 0.4], q=[0.5, 0.5]),
        ),
        (
            [[1.0, -1.0], [-1.0, 1.0]],
            dict(k=2.0, value=0.0, p=[0.5, 0.5], q=[0.5, 0.5]),
        ),
        (
            [[4.0, 1.0], [2.0, 3.0]],
            dict(k=0.0, value=2.5, p=[0.25, 0.75], q=[0.5, 0.5]),
        ),
        (
            [[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]],
            dict(
                k=2.0,
                value=0.0,
                p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                q=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ),
        ),
        (
            [[1.0, 2.0], [3.0, -1.0], [0.0, 4.0]],
            dict(k=2.0, value=1.5, p=[0.0, 0.5, 0.5], q=[0.625, 0.375]),
        ),
        (
            [[4.0, 1.0, -3.0], [3.0, 2.0, 5.0]],
            dict(k=4.0, value=2.0, p=[0.0, 1.0], q=[0.0, 1.0, 0.0]),
        ),
        (
            [[-4.0]],
            dict(k=5.0, value=-4.0, p=[1.0], q=[1.0]),
        ),
    )
    for matrix, expected in cases:
        solution = solve_zero_sum_game(np.array(matrix, dtype=float))
        _assert_consistent(solution)
        if solution.translated.k != expected["k"]:
            raise AssertionError((matrix, solution.translated.k, expected["k"]))
        np.testing.assert_allclose(solution.v, expected["value"], atol=1e-7)
        np.testing.assert_allclose(solution.p, expected["p"], atol=1e-6)
        np.testing.assert_allclose(solution.q, expected["q"], atol=1e-6)
        if not np.all(solution.translated.A_prime > 0):
            raise AssertionError("A' debe ser estrictamente positiva.")
        if solution.method != "highs-ds":
            raise AssertionError(solution.method)
        _assert_dimensions(np.asarray(matrix, dtype=float), solution)

    for row_count, column_count in ((1, 1), (1, 4), (3, 2), (4, 5), (2, 2)):
        zero_game = solve_zero_sum_game(np.zeros((row_count, column_count)))
        _assert_consistent(zero_game)
        _assert_dimensions(np.zeros((row_count, column_count)), zero_game)
        if zero_game.translated.k != 1.0 or abs(zero_game.v) > 1e-8:
            raise AssertionError(
                f"Una matriz nula {row_count}×{column_count} debe trasladarse con k = 1 y valor 0."
            )

    try:
        solve_zero_sum_game(np.array([[np.nan, 1.0], [0.0, 2.0]]))
    except InvalidPayoffError:
        pass
    else:
        raise AssertionError("Una celda vacía o no finita debe rechazarse.")

    try:
        _optimize_primal(np.zeros((2, 2)))
    except InfeasibleProblemError:
        pass
    else:
        raise AssertionError("El primal con pagos nulos debe declararse infactible.")

    try:
        _raise_solver_status(type("Result", (), {"status": 3, "message": "unbounded"})())
    except UnboundedProblemError:
        pass
    else:
        raise AssertionError("El estado no acotado debe tener su propio error.")


if __name__ == "__main__":
    run_self_check()
    demo = solve_zero_sum_game(np.array([[3.0, -1.0], [-2.0, 4.0]]))
    print(f"m = {demo.p.size}, n = {demo.q.size}")
    print("k =", demo.translated.k)
    print("W* =", demo.W)
    print("v* =", demo.v)
    print("p* =", demo.p)
    print("q* =", demo.q)
    print("Comprobaciones correctas.")
