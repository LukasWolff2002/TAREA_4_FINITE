import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def leer_resultados_por_R(path):
    """
    Lee resultados del archivo y agrupa los errores por R. Para cada R, se obtiene una lista de (N, max_error).
    """
    resultados_por_R = defaultdict(list)

    with open(path, "r") as f:
        lines = f.readlines()

    for i in range(0, len(lines), 3):  # Cada bloque tiene 5 líneas
        linea_params = lines[i].strip()
        linea_max_error = lines[i+1].strip()

        n = int(linea_params.split(",")[0].split("=")[1].strip())
        r = float(linea_params.split(",")[1].split("=")[1].strip())
        max_error = float(linea_max_error.split(":")[1].strip())

        resultados_por_R[r].append((n, max_error))

    return resultados_por_R


def graficar_errores_por_R(resultados_por_R):
    """
    Grafica el error máximo en función de N para cada R constante, con eje Y en escala logarítmica.
    """
    plt.figure(figsize=(10, 6))

    for r, valores in sorted(resultados_por_R.items()):
        valores_ordenados = sorted(valores, key=lambda x: x[0])  # ordenar por N
        N_vals = [n for n, _ in valores_ordenados]
        errores = [e for _, e in valores_ordenados]
        plt.plot(N_vals, errores, label=f"R = {r:.2f}")

    plt.xlabel("N (divisions)")
    plt.ylabel("Mean Error (log scale)")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Maximum error vs N for each constant R")
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(title="Progression R", fontsize=9)
    plt.tight_layout()


    plt.savefig("INFORME/GRAFICOS/LST/errores_por_R.png", dpi=300)


# Uso
ruta = "LST/resultados.txt"
resultados_por_R = leer_resultados_por_R(ruta)
graficar_errores_por_R(resultados_por_R)
