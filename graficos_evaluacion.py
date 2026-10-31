"""
graficos_evaluacion.py
Genera los gráficos de la evaluación a partir de la carpeta resultados/.
    python graficos_evaluacion.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(__file__).parent.parent / "resultados"
ANIO = 2024
SUP, TXT, TXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
MEJOR, PEOR, NEUTRO = "#2a78d6", "#eb6834", "#9a9893"
AZULES = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({"figure.facecolor": SUP, "axes.facecolor": SUP, "axes.edgecolor": GRID,
                     "text.color": TXT, "axes.labelcolor": TXT2, "xtick.color": TXT2,
                     "ytick.color": TXT, "font.size": 10, "axes.titleweight": "bold"})


SIGLAS = {"Random Forest": "RF", "Gradient Boosting": "GB", "Holt-Winters": "HW"}


def nombre_corto(m):
    if m.startswith("Ensemble promedio ("):
        partes = m[len("Ensemble promedio ("):-1].split(" + ")
        return "Ensemble " + " + ".join(SIGLAS.get(p, p) for p in partes)
    return (m.replace("Ensemble promedio top 3", "Ensemble top 3 (promedio)")
             .replace("Ensemble ponderado top 3", "Ensemble top 3 (ponderado)"))


def grafico_bosque(t, elegido):
    """Diferencia de error contra la línea base, con intervalo de confianza del 95%."""
    t = t.drop(index="Ingenuo estacional").sort_values("Dif_vs_base_%", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    for i, (m, r) in enumerate(t.iterrows()):
        if r["IC95_sup_%"] < 0:
            c, et = MEJOR, "mejor que la base"
        elif r["IC95_inf_%"] > 0:
            c, et = PEOR, "peor que la base"
        else:
            c, et = NEUTRO, "no se distingue"
        ax.plot([r["IC95_inf_%"], r["IC95_sup_%"]], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.scatter(r["Dif_vs_base_%"], i, s=70, color=c, zorder=3, edgecolor=SUP, linewidth=2)
        ax.text(max(r["IC95_sup_%"], 0) + 0.6, i, f"{r['Dif_vs_base_%']:+.1f}%  ·  {et}",
                va="center", fontsize=8.5, color=TXT2)
    ax.axvline(0, color=TXT2, lw=1, ls=(0, (3, 3)))
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([nombre_corto(m) + ("  ★" if m == elegido else "") for m in t.index])
    ax.set_xlabel("Diferencia de error frente a la línea base (puntos %, escala log)  ·  izquierda = mejor")
    ax.set_xlim(min(-9, t["IC95_inf_%"].min() - 1), t["IC95_sup_%"].max() + 13)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"¿Le ganan los modelos a «el mismo mes del año pasado»?  Prueba {ANIO}", loc="left", pad=12)
    fig.text(0.01, 0.01, "Punto: diferencia media. Línea: intervalo de confianza del 95% (bootstrap pareado por mes, "
                         "5000 réplicas).  ★ = elegido de antemano por la validación.", fontsize=7.5, color=TXT2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RES / "1_modelos_vs_linea_base.png")
    plt.close(fig)


def grafico_mapa(tm, orden):
    """Error porcentual (MAPE) por modalidad y modelo."""
    p = tm.pivot(index="ESTUDIO", columns="Modelo", values="MAPE_%")[orden]
    p = p.loc[p["Ingenuo estacional"].sort_values().index]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("azul", AZULES[1:7])
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=160)
    vmax = np.percentile(p.values, 92)
    ax.imshow(np.clip(p.values, 0, vmax), cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    for i in range(p.shape[0]):
        fila_min = p.iloc[i].min()
        for j in range(p.shape[1]):
            v = p.iat[i, j]
            oscuro = min(v, vmax) / vmax > 0.55
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if oscuro else TXT,
                    fontweight="bold" if v == fila_min else "normal")
    ax.set_xticks(range(p.shape[1]))
    ax.set_xticklabels([nombre_corto(c) for c in p.columns], rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(p.shape[0]))
    ax.set_yticklabels(p.index)
    ax.set_xticks(np.arange(-.5, p.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, p.shape[0]), minor=True)
    ax.grid(which="minor", color=SUP, lw=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Error medio porcentual (MAPE) por modalidad en {ANIO}  ·  en negrita, el mejor de cada fila",
                 loc="left", pad=12)
    fig.text(0.01, 0.01, "Más oscuro = más error. Filas ordenadas por el error de la línea base: abajo, las series más ruidosas.", fontsize=7.5, color=TXT2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RES / "2_error_por_modalidad.png")
    plt.close(fig)


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) > 1:
        RES = Path(sys.argv[1])
    params = json.loads((RES / "mejores_parametros.json").read_text())
    ANIO = params["anio_prueba"]
    t = pd.read_csv(RES / "metricas_globales.csv", index_col=0)
    tm = pd.read_csv(RES / "metricas_por_modalidad.csv")
    elegido = json.loads((RES / "mejores_parametros.json").read_text())["modelo_elegido_por_validacion"]
    grafico_bosque(t, elegido)
    base = ["Ingenuo estacional", "Holt-Winters", "Ridge", "SVR", "Gradient Boosting", "Random Forest"]
    presentes = list(tm["Modelo"].unique())
    orden = [m for m in base if m in presentes] + sorted(m for m in presentes if m.startswith("Ensemble"))
    grafico_mapa(tm, orden)
    print("Gráficos guardados en", RES)
