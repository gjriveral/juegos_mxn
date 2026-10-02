# Juegos de suma cero m × n

Aplicación web para formular y resolver juegos de suma cero de tamaño arbitrario \(m \times n\) en estrategias mixtas. El pago \(a_{ij}\) es lo que recibe el Jugador 1 cuando elige la fila \(i\) y el Jugador 2 la columna \(j\).

El programa traslada la matriz si algún pago no es estrictamente positivo, plantea el problema del Jugador 1 como un programa lineal y lo resuelve con el simplex dual. Las variables del Jugador 2 se leen de los precios sombra de ese primal. Al final recupera el valor del juego original y las estrategias mixtas óptimas.

## Qué muestra

1. La matriz de pagos, con \(m\) y \(n\) elegidos por el usuario.
2. La constante de traslación \(k\) y la matriz \(A'\).
3. El primal: minimizar \(W = \sum x_i\), con una restricción por columna.
4. El dual: maximizar \(Z = \sum y_j\), con \(y_j^*\) tomado de los precios sombra.
5. El valor trasladado \(v'^* = 1/W^*\), el valor original \(v^* = v'^* - k\) y las probabilidades \(p^*\) y \(q^*\).

Los resultados aparecen al pulsar **Resolver juego**.

## Ejecución local

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

La aplicación queda en [http://localhost:8501](http://localhost:8501).
