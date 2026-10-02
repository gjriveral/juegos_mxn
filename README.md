# Juegos de suma cero m × n

Aplicación web para formular y resolver juegos de suma cero de tamaño arbitrario m × n en estrategias mixtas. El pago a<sub>ij</sub> es lo que recibe el Jugador 1 cuando elige la fila i y el Jugador 2 la columna j.

El programa traslada la matriz cuando algún pago no es estrictamente positivo, plantea el problema del Jugador 1 como un programa lineal y lo resuelve con el simplex dual. Las variables del Jugador 2 se leen de los precios sombra de ese primal. Al final recupera el valor del juego original y las estrategias mixtas óptimas.

## Qué muestra

1. La matriz de pagos, con m y n elegidos por el usuario.
2. La constante de traslación k y la matriz trasladada A′.
3. El primal: minimizar W = ∑<sub>i</sub> x<sub>i</sub>, con una restricción por columna.
4. El dual: maximizar Z = ∑<sub>j</sub> y<sub>j</sub>, con y<sub>j</sub><sup>*</sup> tomado de los precios sombra.
5. El valor trasladado v′<sup>*</sup> = 1/W<sup>*</sup>, el valor original v<sup>*</sup> = v′<sup>*</sup> − k, y las probabilidades p<sup>*</sup> y q<sup>*</sup>.

Los resultados aparecen al pulsar **Resolver juego**.

## Ejecución local

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

La aplicación queda en [http://localhost:8501](http://localhost:8501).
