# Ejecutar desde esta carpeta: streamlit run app.py

"""Interfaz para formular y resolver juegos de suma cero m × n."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import formatting as tex
from solver import (
    GameSolution,
    GameSolverError,
    InvalidPayoffError,
    TranslatedGame,
    solve_translated_game,
    translate_payoff,
)

EXERCISE_NAME = "Ejercicio 3×3"
EXAMPLES = {
    "2×2 clásico": np.array([[3.0, -1.0], [-2.0, 4.0]]),
    "3×3 simétrico": np.array(
        [
            [2.0, -1.0, -1.0],
            [-1.0, 2.0, -1.0],
            [-1.0, -1.0, 2.0],
        ]
    ),
    EXERCISE_NAME: np.array(
        [
            [4.0, -2.0, 3.0],
            [2.0, 3.0, -1.0],
            [-2.0, -1.0, 5.0],
        ]
    ),
}
METHOD_LABELS = {
    "highs-ds": "simplex dual de HiGHS",
    "highs": "HiGHS",
}


def main() -> None:
    st.set_page_config(
        page_title="Estrategias mixtas",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Juegos de suma cero en estrategias mixtas")
    st.markdown(
        r"Formula y resuelve un juego matricial $m \times n$ con programación lineal. "
        r"El Jugador 1 elige la fila, el Jugador 2 elige la columna y $a_{ij}$ es el "
        r"pago que recibe el Jugador 1."
    )

    ensure_state()
    rows, columns = configure_sidebar()
    matrix = render_payoff_editor(rows, columns)
    if matrix.shape != (rows, columns):
        st.session_state.show_solution = False
        st.error("La matriz editada no tiene las dimensiones seleccionadas.")
        return

    if payoff_changed(matrix):
        st.session_state.show_solution = False

    solve = st.button(
        "Resolver juego",
        type="primary",
        width="stretch",
        help="Ejecuta la traslación, el primal, el dual y la recuperación de p*, q* y v*.",
    )
    if solve:
        st.session_state.show_solution = True

    if not st.session_state.show_solution:
        st.info(
            "Define m, n y la matriz de pagos. Después pulsa Resolver juego "
            "para obtener todos los resultados."
        )
        return

    try:
        translated = translate_payoff(matrix)
    except InvalidPayoffError as error:
        st.warning(str(error))
        return

    render_translation(translated)
    try:
        solution = solve_translated_game(translated)
    except GameSolverError as error:
        render_primal(translated, None)
        render_dual(translated, None)
        st.error(str(error))
        return

    render_primal(translated, solution)
    render_dual(translated, solution)
    render_recovery(solution)


def ensure_state() -> None:
    if "m" not in st.session_state:
        st.session_state.m = 2
    if "n" not in st.session_state:
        st.session_state.n = 2
    if "example_choice" not in st.session_state:
        st.session_state.example_choice = "Personalizada"
    if "editor_version" not in st.session_state:
        st.session_state.editor_version = 0
    if "show_solution" not in st.session_state:
        st.session_state.show_solution = False
    if "payoff_signature" not in st.session_state:
        st.session_state.payoff_signature = None
    if "matrix" not in st.session_state:
        st.session_state.matrix = zero_payoff_frame(
            int(st.session_state.m),
            int(st.session_state.n),
        )
    if st.session_state.get("active_exercise") != EXERCISE_NAME:
        exercise = EXAMPLES[EXERCISE_NAME]
        row_count, column_count = exercise.shape
        st.session_state.m = int(row_count)
        st.session_state.n = int(column_count)
        st.session_state.example_choice = EXERCISE_NAME
        st.session_state.matrix = payoff_frame(row_count, column_count, exercise)
        st.session_state.editor_version = int(st.session_state.editor_version) + 1
        st.session_state.active_exercise = EXERCISE_NAME


def configure_sidebar() -> tuple[int, int]:
    st.sidebar.header("Datos del juego")
    st.sidebar.selectbox(
        "Ejemplo",
        options=["Personalizada", *EXAMPLES.keys()],
        key="example_choice",
        on_change=load_example,
        help="Al elegir un ejemplo se reemplaza la matriz actual.",
    )
    rows = int(
        st.sidebar.number_input(
            "m — estrategias del Jugador 1 (filas)",
            min_value=1,
            max_value=10,
            step=1,
            key="m",
            on_change=resize_matrix,
        )
    )
    columns = int(
        st.sidebar.number_input(
            "n — estrategias del Jugador 2 (columnas)",
            min_value=1,
            max_value=10,
            step=1,
            key="n",
            on_change=resize_matrix,
        )
    )
    st.sidebar.button("Vaciar matriz", on_click=clear_matrix)
    st.sidebar.caption("Cada celda es el pago que recibe el Jugador 1.")
    return rows, columns


def load_example() -> None:
    matrix = EXAMPLES.get(st.session_state.example_choice)
    if matrix is None:
        return
    row_count, column_count = matrix.shape
    st.session_state.m = int(row_count)
    st.session_state.n = int(column_count)
    st.session_state.matrix = payoff_frame(row_count, column_count, matrix)
    st.session_state.editor_version += 1
    st.session_state.show_solution = False


def resize_matrix() -> None:
    st.session_state.example_choice = "Personalizada"
    replace_matrix(int(st.session_state.m), int(st.session_state.n))
    st.session_state.show_solution = False


def clear_matrix() -> None:
    st.session_state.example_choice = "Personalizada"
    replace_matrix(int(st.session_state.m), int(st.session_state.n))
    st.session_state.show_solution = False


def replace_matrix(row_count: int, column_count: int, values: np.ndarray | None = None) -> None:
    st.session_state.matrix = payoff_frame(row_count, column_count, values)
    st.session_state.editor_version += 1


def mark_matrix_custom() -> None:
    st.session_state.example_choice = "Personalizada"


def render_payoff_editor(rows: int, columns: int) -> np.ndarray:
    current = st.session_state.matrix
    if not isinstance(current, pd.DataFrame) or tuple(current.shape) != (rows, columns):
        replace_matrix(rows, columns)

    frame = st.session_state.matrix
    with st.container(border=True):
        st.subheader("1. Matriz de pagos")
        st.caption(
            "Las filas son las estrategias del Jugador 1 y las columnas las del "
            "Jugador 2. Cuando la matriz esté lista, pulsa Resolver juego."
        )
        edited = st.data_editor(
            frame,
            key=editor_key(rows, columns),
            num_rows="fixed",
            hide_index=False,
            on_change=mark_matrix_custom,
            column_config={
                column: st.column_config.NumberColumn(
                    column,
                    help="Pago para el Jugador 1",
                    step=0.5,
                    format="%.4f",
                )
                for column in frame.columns
            },
        )
    values = edited.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    st.session_state.matrix = payoff_frame(rows, columns, values)
    return values


def render_translation(game: TranslatedGame) -> None:
    with st.container(border=True):
        st.subheader("2. Traslación de escala")
        st.markdown(_translation_reason(game.minimum))
        metric_k, metric_min = st.columns(2)
        metric_k.metric("Constante k", tex.display_number(game.k))
        metric_min.metric("Pago mínimo de A", tex.display_number(game.minimum))
        st.latex(tex.translation_latex(game.minimum, game.k))
        st.latex(tex.cell_shift_latex())

        original, shifted = st.columns(2)
        with original:
            st.markdown("**Matriz original A**")
            show_matrix(game.A)
        with shifted:
            st.markdown("**Matriz trasladada A'**")
            show_matrix(game.A_prime)
        if game.A.shape[0] <= 6 and game.A.shape[1] <= 6:
            st.latex(tex.matrix_latex("A'", game.A_prime))


def render_primal(game: TranslatedGame, solution: GameSolution | None) -> None:
    rows, columns = game.A_prime.shape
    with st.container(border=True):
        st.subheader("3. Problema primal: Jugador 1")
        st.markdown(
            r"El Jugador 1 maximiza el valor del juego. Con pagos estrictamente "
            r"positivos, $v^{\prime *} > 0$ y el cambio de variable siguiente convierte "
            r"ese objetivo en minimizar la suma de las $x_i$."
        )
        st.latex(tex.primal_substitution_latex())
        st.latex(tex.primal_symbolic_latex(rows, columns))
        st.markdown("**Problema con los coeficientes de A'**")
        st.latex(tex.primal_numeric_latex(game.A_prime))
        if solution is None:
            return
        st.latex(tex.optimum_latex("x", solution.x, "W", solution.W))
        table, metric = st.columns([2, 1])
        with table:
            show_vector(solution.x, "x")
        with metric:
            st.metric("W*", tex.display_number(solution.W))
        method = METHOD_LABELS.get(solution.method, solution.method)
        st.caption(f"Solución de scipy.optimize.linprog con el {method}.")


def render_dual(game: TranslatedGame, solution: GameSolution | None) -> None:
    rows, columns = game.A_prime.shape
    with st.container(border=True):
        st.subheader("4. Problema dual: Jugador 2")
        st.markdown(
            r"El Jugador 2 minimiza el valor del juego. Su modelo es el dual del "
            r"primal: una variable $y_j$ por columna y una restricción por fila."
        )
        st.latex(tex.dual_substitution_latex())
        st.latex(tex.dual_symbolic_latex(rows, columns))
        st.markdown("**Problema con los coeficientes de A'**")
        st.latex(tex.dual_numeric_latex(game.A_prime))
        if solution is None:
            return
        st.markdown(
            r"$y_j^*$ es el precio sombra de la restricción de la columna $j$ del "
            r"primal, $\sum_i a^{\prime}_{ij} x_i \ge 1$. No se resuelve un segundo "
            r"programa lineal: el solver entrega el marginal de la forma $\le$ con la "
            r"que se cargó esa restricción, y $y_j^*$ es el opuesto de ese marginal."
        )
        st.latex(tex.optimum_latex("y", solution.y, "Z", solution.Z))
        table, metric = st.columns([2, 1])
        with table:
            show_vector(solution.y, "y")
        with metric:
            st.metric("Z*", tex.display_number(solution.Z))
        if abs(solution.Z - solution.W) <= 1e-5 * max(1.0, abs(solution.W)):
            st.caption("Por dualidad fuerte, Z* coincide con W*.")
        else:
            st.warning(
                "La suma de los precios sombra no reprodujo W* con la tolerancia esperada."
            )


def render_recovery(solution: GameSolution) -> None:
    with st.container(border=True):
        st.subheader("5. Valor del juego y estrategias óptimas")
        for warning in solution.warnings:
            st.warning(warning)
        st.latex(
            tex.value_recovery_latex(
                solution.W,
                solution.v_prime,
                solution.translated.k,
                solution.v,
            )
        )
        st.latex(
            tex.strategy_recovery_latex(
                solution.x,
                solution.y,
                solution.v_prime,
                solution.p,
                solution.q,
            )
        )

        row_count, column_count = solution.translated.A.shape
        if solution.p.shape != (row_count,) or solution.q.shape != (column_count,):
            st.error(
                "Los vectores de probabilidad no tienen las dimensiones del juego: "
                f"p* debe tener longitud {row_count} y q* longitud {column_count}."
            )
            return

        st.markdown("**Valor esperado del juego**")
        st.latex(tex.V_STAR + " = " + tex.latex_number(solution.v))
        st.metric("v*", tex.display_number(solution.v))
        st.caption(
            "v* = v'* − k, con v'* = "
            + tex.display_number(solution.v_prime)
            + " y k = "
            + tex.display_number(solution.translated.k)
            + "."
        )

        st.markdown("**Probabilidades del Jugador 1**")
        st.text(format_vector(solution.p))
        for index, probability in enumerate(solution.p, start=1):
            st.markdown(
                f"Estrategia {index}: $p_{{{index}}} = {tex.latex_number(float(probability))}$"
            )

        st.markdown("**Probabilidades del Jugador 2**")
        st.text(format_vector(solution.q))
        for index, probability in enumerate(solution.q, start=1):
            st.markdown(
                f"Estrategia {index}: $q_{{{index}}} = {tex.latex_number(float(probability))}$"
            )

        expected = float(solution.p @ solution.translated.A @ solution.q)
        if abs(expected - solution.v) > 1e-4:
            st.warning(
                "La comprobación p*^T A q* = "
                + tex.display_number(expected)
                + " difiere de v* = "
                + tex.display_number(solution.v)
                + ". Revisa la escala de los pagos."
            )
        else:
            st.caption(
                "Comprobación: el pago esperado p*^T A q* es "
                + tex.display_number(expected)
                + ". Si hay varias estrategias óptimas, el solver devuelve una de ellas."
            )


def payoff_changed(matrix: np.ndarray) -> bool:
    signature = (matrix.shape, np.array(matrix, dtype=float).tobytes())
    changed = st.session_state.payoff_signature not in (None, signature)
    st.session_state.payoff_signature = signature
    return changed


def strategy_labels(count: int) -> list[str]:
    return [f"Estrategia {index}" for index in range(1, count + 1)]


def payoff_frame(
    row_count: int,
    column_count: int,
    values: np.ndarray | None = None,
) -> pd.DataFrame:
    """DataFrame m × n. Sin valores, queda inicializado en ceros."""

    if values is None:
        data = np.zeros((row_count, column_count), dtype=float)
    else:
        data = np.asarray(values, dtype=float)
        if data.shape != (row_count, column_count):
            raise ValueError("Los pagos no coinciden con las dimensiones m y n.")
    frame = pd.DataFrame(
        data,
        index=strategy_labels(row_count),
        columns=strategy_labels(column_count),
    )
    frame.index.name = "Jugador 1"
    frame.columns.name = "Jugador 2"
    return frame


def zero_payoff_frame(row_count: int, column_count: int) -> pd.DataFrame:
    return payoff_frame(row_count, column_count)


def editor_key(rows: int, columns: int) -> str:
    return f"payoff_{rows}_{columns}_{st.session_state.editor_version}"


def show_matrix(values: np.ndarray) -> None:
    row_count, column_count = values.shape
    frame = payoff_frame(row_count, column_count, values).map(
        lambda entry: tex.display_number(float(entry))
    )
    st.dataframe(frame, hide_index=False)


def show_vector(values: np.ndarray, symbol: str) -> None:
    frame = pd.DataFrame(
        {symbol: [tex.display_number(float(component)) for component in values]},
        index=strategy_labels(len(values)),
    )
    frame.index.name = "Jugador 1" if symbol in {"x", "p"} else "Jugador 2"
    st.dataframe(frame, hide_index=False)


def format_vector(values: np.ndarray) -> str:
    components = ", ".join(
        tex.display_number(float(component)) for component in np.asarray(values, dtype=float)
    )
    return f"[{components}]"


def _translation_reason(minimum: float) -> str:
    if minimum < 0:
        return (
            "Hay pagos negativos. Se elige una constante k que lleva el pago más "
            "pequeño a 1, de modo que toda entrada de A' queda estrictamente positiva. "
            "El valor del juego original se recupera restando esa misma constante."
        )
    if minimum == 0:
        return (
            "No hay pagos negativos, pero sí pagos nulos. El cambio de variable exige "
            "entradas estrictamente positivas, así que también se traslada la matriz "
            "hasta que el pago más pequeño sea 1."
        )
    return (
        "Todos los pagos ya son estrictamente positivos. Se toma k = 0 y la matriz "
        "trasladada coincide con la original."
    )


main()
