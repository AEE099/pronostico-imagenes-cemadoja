"""
evaluacion_modelos.py
Evaluación rigurosa de modelos de predicción de demanda mensual por modalidad.

Diseño (pensado para evitar fuga de información):
  1. El último año (por defecto 2024) queda SELLADO como prueba final.
  2. Los hiperparámetros se buscan con validación cruzada TEMPORAL dentro del
     periodo de entrenamiento: se entrena con el pasado y se valida con los
     meses siguientes (ventana expansiva), nunca al revés.
  3. El ensemble se arma con las predicciones fuera de muestra de la
     validación, no mirando la prueba final.
  4. Con los mejores parámetros se reentrena con todo el entrenamiento y se
     predice el año de prueba completo (horizonte de 1 a 12 meses).

Modelos:
  - Ingenuo estacional (línea base): el mismo mes del año anterior.
  - Holt-Winters aditivo (suavizado exponencial), por modalidad.
  - Ridge (regresión lineal regularizada), con o sin interacciones.
  - Random Forest, Gradient Boosting y SVR.
  - Ensembles: promedio de los 2 y 3 mejores, y promedio ponderado.

Los modelos de ML son GLOBALES (aprenden de todas las modalidades a la vez) y
trabajan sobre log(1 + cantidad), para que una modalidad grande como Rayos X no
domine el aprendizaje y los errores se interpreten como proporcionales.

Uso:
    python analisis/evaluacion_modelos.py --tratar-interrupciones
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
SEMILLA = 42
BASE_NOMBRE = "Ingenuo estacional"
COLS_X = ["ESTUDIO", "MES_NUM", "T"]


# =============================================================================
# Datos
# =============================================================================

def cargar(ruta) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    df["FECHA"] = pd.to_datetime(dict(year=df["ANO"], month=df["MES_NUM"], day=1))
    inicio = df["FECHA"].min()
    # T = meses transcurridos desde el inicio (captura la tendencia)
    df["T"] = (df["FECHA"].dt.year - inicio.year) * 12 + (df["FECHA"].dt.month - inicio.month)
    return df.sort_values(["FECHA", "ESTUDIO"]).reset_index(drop=True)


def tratar_interrupciones(datos: pd.DataFrame, umbral=0.25, ventana=12):
    """Detecta interrupciones del servicio (COVID, equipos parados) de forma CAUSAL:
    un mes es interrupción si cae por debajo del `umbral` (25%) de la mediana de los
    `ventana` meses ANTERIORES de su modalidad. Se reemplaza por esa mediana.

    Al usar solo meses pasados, tratar la serie completa da el mismo resultado que
    tratarla dentro de cada ronda de validación: no entra información del futuro.
    Devuelve una copia con CANTIDAD tratada y la columna INTERRUPCION (bool)."""
    datos = datos.copy()
    datos["INTERRUPCION"] = False
    marcados = []
    for est, g in datos.groupby("ESTUDIO"):
        g = g.sort_values("FECHA")
        vals = g["CANTIDAD"].astype(float).to_numpy().copy()
        for i in range(len(vals)):
            previos = vals[max(0, i - ventana):i]  # ya tratados: interrupciones seguidas se detectan
            if len(previos) < 6:
                continue
            med = np.median(previos)
            if med > 0 and vals[i] < umbral * med:
                vals[i] = round(med)
                fila = g.index[i]
                datos.at[fila, "INTERRUPCION"] = True
                marcados.append(f"{est} {g.at[fila, 'FECHA']:%Y-%m}")
        datos.loc[g.index, "CANTIDAD"] = vals
    return datos, marcados


def particiones_temporales(train: pd.DataFrame, n_bloques=4, meses_bloque=3, excluir=None):
    """Validación cruzada con ventana expansiva sobre los últimos meses del entrenamiento.
    Ej. con 2019–2025 y 3 bloques de 12: aprender 2019–2022 y validar 2023; aprender
    2019–2023 y validar 2024; aprender 2019–2024 y validar 2025.
    `excluir` (máscara booleana) saca filas de la CALIFICACIÓN de validación, por
    ejemplo meses de interrupción, que ningún modelo puede predecir."""
    fechas = np.sort(train["FECHA"].unique())
    inicio_val = len(fechas) - n_bloques * meses_bloque
    if inicio_val < 12:
        raise ValueError("Se necesitan al menos 12 meses de entrenamiento antes del primer bloque.")
    fuera = np.zeros(len(train), bool) if excluir is None else np.asarray(excluir, bool)
    splits = []
    for k in range(n_bloques):
        a, b = inicio_val + k * meses_bloque, inicio_val + (k + 1) * meses_bloque
        tr = np.where(train["FECHA"] < fechas[a])[0]
        va = np.where((train["FECHA"] >= fechas[a]) & (train["FECHA"] <= fechas[b - 1]) & ~fuera)[0]
        splits.append((tr, va))
    return splits


# =============================================================================
# Métricas
# =============================================================================

def metricas(y, p, escala_mase=None) -> dict:
    """y, p en escala original. escala_mase: MAE del ingenuo estacional dentro del entrenamiento."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    pos = y > 0
    m = {
        "MAE": np.mean(np.abs(e)),
        "MSE": np.mean(e ** 2),
        "RMSE": np.sqrt(np.mean(e ** 2)),
        "MAPE_%": np.mean(np.abs(e[pos]) / y[pos]) * 100,
        "sMAPE_%": np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p) + 1e-9)) * 100,
        "WAPE_%": np.sum(np.abs(e)) / np.sum(np.abs(y)) * 100,
        "Sesgo_%": np.sum(e) / np.sum(y) * 100,  # >0 sobreestima, <0 subestima
        "R2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
    }
    if escala_mase is not None:
        m["MASE"] = m["MAE"] / escala_mase
    return m


def _mae_log(y, p):
    return np.mean(np.abs(np.log1p(y) - np.log1p(np.clip(p, 0, None))))


# Puntuación usada para elegir hiperparámetros: error absoluto medio en escala log.
# Equivale aproximadamente a un error porcentual y trata igual a todas las modalidades.
PUNTAJE = make_scorer(_mae_log, greater_is_better=False)


# =============================================================================
# Modelos de series temporales (por modalidad)
# =============================================================================

def ingenuo_estacional(hist: pd.DataFrame, futuro: pd.DataFrame) -> np.ndarray:
    """Valor del mismo mes, 12 meses antes. Para horizontes >12 se repite el último año."""
    serie = hist.set_index("FECHA")["CANTIDAD"]
    ult = hist["FECHA"].max()
    pred = []
    for f in futuro["FECHA"]:
        ref = f
        while ref > ult:
            ref = ref - pd.DateOffset(years=1)
        pred.append(serie.get(ref, np.nan))
    return np.array(pred, float)


class HoltWinters:
    """Holt-Winters aditivo con estacionalidad de 12 meses, sobre log(1 + y).
    alpha, beta, gamma (y phi si la tendencia es amortiguada) se ajustan
    minimizando el error de un paso adelante."""

    def __init__(self, tendencia="add"):
        assert tendencia in ("none", "add", "damped")
        self.tendencia = tendencia

    def _filtrar(self, y, a, b, g, phi):
        m = 12
        nivel = y[:m].mean()
        tend = (y[m:2 * m].mean() - y[:m].mean()) / m if (len(y) >= 2 * m and self.tendencia != "none") else 0.0
        est = list(y[:m] - nivel)
        errores = []
        for t in range(m, len(y)):
            s = est[t - m]
            pron = nivel + phi * tend + s
            errores.append(y[t] - pron)
            nivel_ant = nivel
            nivel = a * (y[t] - s) + (1 - a) * (nivel + phi * tend)
            tend = b * (nivel - nivel_ant) + (1 - b) * phi * tend if self.tendencia != "none" else 0.0
            est.append(g * (y[t] - nivel) + (1 - g) * s)
        return np.array(errores), nivel, tend, est

    def fit(self, y):
        y = np.log1p(np.asarray(y, float))
        if self.tendencia == "damped":
            x0, lim = [0.3, 0.05, 0.1, 0.9], [(0, 1), (0, 1), (0, 1), (0.8, 0.98)]
        else:
            x0, lim = [0.3, 0.05, 0.1], [(0, 1), (0, 1), (0, 1)]

        def sse(x):
            a, b, g = x[:3]
            phi = x[3] if self.tendencia == "damped" else 1.0
            e, *_ = self._filtrar(y, a, b, g, phi)
            return np.sum(e ** 2) if len(e) else 0.0

        r = minimize(sse, x0, bounds=lim, method="L-BFGS-B")
        self.params_ = dict(zip(["alpha", "beta", "gamma", "phi"], list(r.x) + ([] if self.tendencia == "damped" else [1.0])))
        _, self.nivel_, self.tend_, self.est_ = self._filtrar(y, *[self.params_[k] for k in ["alpha", "beta", "gamma", "phi"]])
        return self

    def predict(self, h):
        phi = self.params_["phi"]
        n = len(self.est_)
        out = []
        for i in range(1, h + 1):
            amort = sum(phi ** j for j in range(1, i + 1))
            s = self.est_[n - 12 + (i - 1) % 12]
            out.append(self.nivel_ + amort * self.tend_ + s)
        return np.clip(np.expm1(np.array(out)), 0, None)


def holt_winters_predecir(hist, futuro, tendencia):
    """Ajusta un Holt-Winters por modalidad y predice las fechas de `futuro`."""
    pred = pd.Series(index=futuro.index, dtype=float)
    for est, fut in futuro.groupby("ESTUDIO"):
        h = hist[hist["ESTUDIO"] == est].sort_values("FECHA")
        pasos = ((fut["FECHA"].dt.year - h["FECHA"].max().year) * 12
                 + fut["FECHA"].dt.month - h["FECHA"].max().month).to_numpy()
        hw = HoltWinters(tendencia).fit(h["CANTIDAD"].to_numpy())
        pred.loc[fut.index] = hw.predict(pasos.max())[pasos - 1]
    return pred.to_numpy()


# =============================================================================
# Modelos de machine learning (globales)
# =============================================================================

def preprocesador(escalar_t=False):
    t = StandardScaler() if escalar_t else "passthrough"
    return ColumnTransformer([
        ("estudio", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["ESTUDIO"]),
        ("mes", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["MES_NUM"]),
        ("t", t, ["T"]),
    ])


def con_log(modelo):
    """Entrena sobre log(1+y) y devuelve predicciones en escala original."""
    return TransformedTargetRegressor(regressor=modelo, func=np.log1p, inverse_func=np.expm1)


def espacios_busqueda():
    """(nombre, estimador, espacio de parámetros, tipo de búsqueda, n_iter)."""
    r = "regressor__"
    return [
        ("Ridge",
         con_log(Pipeline([("pre", preprocesador(escalar_t=True)),
                           ("inter", "passthrough"),
                           ("m", Ridge())])),
         {f"{r}inter": ["passthrough",
                        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)],
          f"{r}m__alpha": np.logspace(-3, 3, 13)},
         "grid", None),

        ("Random Forest",
         con_log(Pipeline([("pre", preprocesador()),
                           ("m", RandomForestRegressor(random_state=SEMILLA, n_jobs=1))])),
         {f"{r}m__n_estimators": randint(100, 600),
          f"{r}m__max_depth": [None, 4, 6, 8, 12, 16],
          f"{r}m__min_samples_leaf": randint(1, 8),
          f"{r}m__max_features": [1.0, 0.8, 0.6, 0.4, "sqrt"]},
         "random", 60),

        ("Gradient Boosting",
         con_log(Pipeline([("pre", preprocesador()),
                           ("m", GradientBoostingRegressor(random_state=SEMILLA))])),
         {f"{r}m__n_estimators": randint(50, 800),
          f"{r}m__learning_rate": loguniform(0.005, 0.3),
          f"{r}m__max_depth": randint(2, 7),
          f"{r}m__subsample": uniform(0.5, 0.5),
          f"{r}m__min_samples_leaf": randint(1, 10)},
         "random", 80),

        ("SVR",
         con_log(Pipeline([("pre", preprocesador(escalar_t=True)), ("m", SVR())])),
         [{f"{r}m__kernel": ["rbf"], f"{r}m__C": np.logspace(-1, 3, 9),
           f"{r}m__gamma": ["scale", 0.01, 0.03, 0.1, 0.3], f"{r}m__epsilon": [0.01, 0.03, 0.1]},
          {f"{r}m__kernel": ["linear"], f"{r}m__C": np.logspace(-2, 2, 9),
           f"{r}m__epsilon": [0.01, 0.03, 0.1]}],
         "grid", None),
    ]


def _limpiar_params(p):
    out = {}
    for k, v in p.items():
        k = k.replace("regressor__", "").replace("m__", "")
        if k == "inter":
            v = "con interacciones" if not isinstance(v, str) else "sin interacciones"
        elif isinstance(v, (np.floating, float)):
            v = float(round(v, 5))
        elif isinstance(v, np.integer):
            v = int(v)
        out[k] = v
    return out


# =============================================================================
# Flujo principal
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=str(Path(__file__).parent.parent / "estudios_cemadoja.csv"))
    ap.add_argument("--anio-prueba", type=int, default=2026)
    ap.add_argument("--salida", default=str(Path(__file__).parent.parent / "resultados"))
    ap.add_argument("--bloques", type=int, default=3, help="Bloques de validación temporal")
    ap.add_argument("--meses-bloque", type=int, default=12, help="Meses por bloque de validación")
    ap.add_argument("--desde", type=int, default=None,
                    help="Primer año usado para entrenar (p. ej. 2021 para dejar fuera el COVID)")
    ap.add_argument("--tratar-interrupciones", action="store_true",
                    help="Reemplaza en el ENTRENAMIENTO los meses con caídas >75%% (la prueba no se toca)")
    ap.add_argument("--horizonte", type=int, default=12, help="Meses a pronosticar al final")
    a = ap.parse_args()
    salida = Path(a.salida)
    salida.mkdir(exist_ok=True)

    df = cargar(a.datos)
    if a.desde:
        df = df[df["ANO"] >= a.desde].reset_index(drop=True)
    train = df[df["ANO"] < a.anio_prueba].reset_index(drop=True)
    test = df[df["ANO"] == a.anio_prueba].reset_index(drop=True)
    excluir = None
    if a.tratar_interrupciones:
        train, marcados = tratar_interrupciones(train)
        excluir = train["INTERRUPCION"].to_numpy()
        print(f"Interrupciones tratadas en el entrenamiento ({len(marcados)}), detección causal "
              f"y fuera de la calificación de validación: {', '.join(marcados)}")
    splits = particiones_temporales(train, a.bloques, a.meses_bloque, excluir)
    print(f"Entrenamiento: {train['FECHA'].min():%b %Y} – {train['FECHA'].max():%b %Y} "
          f"({train['FECHA'].nunique()} meses × {train['ESTUDIO'].nunique()} modalidades)")
    print(f"Prueba sellada: {test['FECHA'].min():%b %Y} – {test['FECHA'].max():%b %Y}")
    print(f"Validación cruzada temporal: {len(splits)} bloques de "
          f"{len(np.unique(train.loc[splits[0][1], 'FECHA']))} meses\n")

    oof = pd.DataFrame({"y": np.nan}, index=train.index)  # predicciones fuera de muestra
    val_idx = np.concatenate([va for _, va in splits])
    oof.loc[val_idx, "y"] = train.loc[val_idx, "CANTIDAD"]
    pred_test = pd.DataFrame({"ESTUDIO": test["ESTUDIO"], "FECHA": test["FECHA"], "y": test["CANTIDAD"]})
    mejores, tiempos, estimadores = {}, {}, {}

    # ---- 1. Línea base ----
    oof["Ingenuo estacional"] = np.nan
    for tr, va in splits:
        oof.loc[va, "Ingenuo estacional"] = ingenuo_estacional_global(train.loc[tr], train.loc[va])
    pred_test["Ingenuo estacional"] = ingenuo_estacional_global(train, test)
    mejores["Ingenuo estacional"] = {"parametros": "ninguno (repite el mismo mes del año anterior)"}

    # ---- 2. Holt-Winters: se elige el tipo de tendencia con la misma validación ----
    t0 = time.time()
    res_hw = {}
    for tend in ["none", "add", "damped"]:
        p = pd.Series(index=val_idx, dtype=float)
        for tr, va in splits:
            p.loc[va] = holt_winters_predecir(train.loc[tr], train.loc[va], tend)
        res_hw[tend] = (_mae_log(train.loc[val_idx, "CANTIDAD"], p.loc[val_idx]), p)
    tend_ok = min(res_hw, key=lambda k: res_hw[k][0])
    oof["Holt-Winters"] = np.nan
    oof.loc[val_idx, "Holt-Winters"] = res_hw[tend_ok][1].loc[val_idx]
    pred_test["Holt-Winters"] = holt_winters_predecir(train, test, tend_ok)
    nombres_t = {"none": "sin tendencia", "add": "aditiva", "damped": "amortiguada"}
    mejores["Holt-Winters"] = {"parametros": {"tendencia": nombres_t[tend_ok],
                                              "estacionalidad": "aditiva, 12 meses (sobre log)"},
                               "cv_mae_log_por_opcion": {nombres_t[k]: round(v[0], 4) for k, v in res_hw.items()}}
    tiempos["Holt-Winters"] = time.time() - t0
    print(f"Holt-Winters: mejor tendencia = {nombres_t[tend_ok]}")

    # ---- 3. Modelos de ML con GridSearch / RandomizedSearch ----
    X_tr, y_tr = train[COLS_X], train["CANTIDAD"]
    for nombre, est, espacio, tipo, n_iter in espacios_busqueda():
        t0 = time.time()
        if tipo == "grid":
            bus = GridSearchCV(est, espacio, cv=splits, scoring=PUNTAJE, n_jobs=-1)
        else:
            bus = RandomizedSearchCV(est, espacio, n_iter=n_iter, cv=splits, scoring=PUNTAJE,
                                     n_jobs=-1, random_state=SEMILLA)
        bus.fit(X_tr, y_tr)
        n_comb = len(bus.cv_results_["params"])
        # predicciones fuera de muestra con los mejores parámetros
        oof[nombre] = np.nan
        for tr, va in splits:
            m = clone(bus.best_estimator_)
            m.fit(X_tr.loc[tr], y_tr.loc[tr])
            oof.loc[va, nombre] = m.predict(X_tr.loc[va])
        pred_test[nombre] = np.clip(bus.best_estimator_.predict(test[COLS_X]), 0, None)
        estimadores[nombre] = bus.best_estimator_
        mejores[nombre] = {"busqueda": f"{'GridSearchCV' if tipo == 'grid' else 'RandomizedSearchCV'}, "
                                       f"{n_comb} combinaciones × {len(splits)} bloques",
                           "parametros": _limpiar_params(bus.best_params_),
                           "cv_mae_log": round(-bus.best_score_, 4)}
        tiempos[nombre] = time.time() - t0
        print(f"{nombre:18s} {n_comb:4d} combinaciones  CV MAE(log) = {-bus.best_score_:.4f}  "
              f"({tiempos[nombre]:.0f} s)")

    # ---- 4. Ensembles, elegidos con la validación (no con la prueba) ----
    individuales = [c for c in oof.columns if c != "y"]
    cv_score = {c: _mae_log(oof.loc[val_idx, "y"], oof.loc[val_idx, c]) for c in individuales}
    candidatos = sorted([c for c in individuales if c != "Ingenuo estacional"], key=cv_score.get)
    top2, top3 = candidatos[:2], candidatos[:3]
    pesos = {c: 1 / cv_score[c] for c in top3}
    tot = sum(pesos.values())
    pesos = {c: w / tot for c, w in pesos.items()}

    ens = {
        f"Ensemble promedio ({' + '.join(top2)})": (top2, {c: 0.5 for c in top2}),
        f"Ensemble promedio top 3": (top3, {c: 1 / 3 for c in top3}),
        f"Ensemble ponderado top 3": (top3, pesos),
    }
    for nombre, (cols, w) in ens.items():
        oof[nombre] = sum(oof[c] * w[c] for c in cols)
        pred_test[nombre] = sum(pred_test[c] * w[c] for c in cols)
        cv_score[nombre] = _mae_log(oof.loc[val_idx, "y"], oof.loc[val_idx, nombre])
        mejores[nombre] = {"componentes": {c: round(w[c], 3) for c in cols},
                           "cv_mae_log": round(cv_score[nombre], 4)}
    print(f"\nEnsembles armados con: {', '.join(top3)}")

    # ---- 4b. Mejor modelo POR MODALIDAD, elegido solo con la validación ----
    # Regla de seguridad fijada de antemano: el modelo elegido debe reducir al menos
    # un 5% el error de validación de la línea base; si no, se usa la línea base.
    MEJORA_MINIMA = 0.05
    filas_val, seleccion = [], {}
    oof_val = oof.loc[val_idx].assign(ESTUDIO=train.loc[val_idx, "ESTUDIO"].to_numpy())
    for est, g in oof_val.groupby("ESTUDIO"):
        punt = {m: _mae_log(g["y"], g[m]) for m in individuales + list(ens)}
        pos = g["y"] > 0
        for m, s in punt.items():
            mape = np.mean(np.abs(g.loc[pos, m] - g.loc[pos, "y"]) / g.loc[pos, "y"]) * 100
            filas_val.append({"ESTUDIO": est, "Modelo": m, "VAL_MAE_log": s, "VAL_MAPE_%": mape})
        mejor = min(punt, key=punt.get)
        if punt[mejor] <= (1 - MEJORA_MINIMA) * punt[BASE_NOMBRE]:
            seleccion[est] = (mejor, "mejor en validación")
        else:
            seleccion[est] = (BASE_NOMBRE, "línea base: ningún modelo la superó con claridad")
    val_mod = pd.DataFrame(filas_val)

    # ---- 5. Evaluación final en el año sellado ----
    modelos = individuales + list(ens)
    elegido_cv = min([m for m in modelos], key=cv_score.get)

    # Escala MASE por modalidad: MAE del ingenuo estacional dentro del entrenamiento
    escala = {}
    for est, g in train.groupby("ESTUDIO"):
        s = g.set_index("FECHA")["CANTIDAD"].sort_index()
        escala[est] = np.mean(np.abs(s.values[12:] - s.values[:-12]))

    filas_glob, filas_mod = [], []
    for m in modelos:
        mod_m = []
        for est, g in pred_test.groupby("ESTUDIO"):
            r = metricas(g["y"], g[m], escala[est])
            mod_m.append(r)
            filas_mod.append({"Modelo": m, "ESTUDIO": est, **r})
        glob = metricas(pred_test["y"], pred_test[m])
        glob["MASE"] = np.mean([r["MASE"] for r in mod_m])  # promedio entre modalidades
        glob["MAPE_%"] = np.mean([r["MAPE_%"] for r in mod_m])  # promedio entre modalidades
        glob["R2_medio_por_modalidad"] = np.mean([r["R2"] for r in mod_m])
        glob["CV_MAE_log"] = cv_score[m]
        filas_glob.append({"Modelo": m, **glob})

    tabla = pd.DataFrame(filas_glob).set_index("Modelo")

    # ---- 6. ¿La diferencia con la línea base es real o es suerte? ----
    # Bootstrap pareado por MES: se remuestrean los 12 meses de prueba (con todas sus
    # modalidades) y se recalcula la diferencia de error en escala log con el ingenuo.
    rng = np.random.default_rng(SEMILLA)
    meses_t = pred_test["FECHA"].unique()
    idx_mes = {f: np.where(pred_test["FECHA"] == f)[0] for f in meses_t}
    err_log = {m: np.abs(np.log1p(pred_test["y"]) - np.log1p(pred_test[m].clip(lower=0))).to_numpy()
               for m in modelos}
    for m in modelos:
        dif = err_log[m] - err_log["Ingenuo estacional"]
        boots = []
        for _ in range(5000):
            sel = np.concatenate([idx_mes[f] for f in rng.choice(meses_t, len(meses_t))])
            boots.append(dif[sel].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        tabla.loc[m, "Dif_vs_base_%"] = dif.mean() * 100
        tabla.loc[m, "IC95_inf_%"] = lo * 100
        tabla.loc[m, "IC95_sup_%"] = hi * 100

    tabla = tabla.sort_values("MASE")
    tabla_mod = pd.DataFrame(filas_mod)

    # ---- 6b. ¿Cómo le fue en la prueba sellada a la selección por modalidad? ----
    tm = tabla_mod.set_index(["ESTUDIO", "Modelo"])
    filas_sel = []
    for est, (m, regla) in seleccion.items():
        filas_sel.append({
            "ESTUDIO": est, "MODELO": m, "REGLA": regla,
            "VAL_MAPE_%": val_mod.set_index(["ESTUDIO", "Modelo"]).loc[(est, m), "VAL_MAPE_%"],
            "VAL_MAPE_BASE_%": val_mod.set_index(["ESTUDIO", "Modelo"]).loc[(est, BASE_NOMBRE), "VAL_MAPE_%"],
            "PRUEBA_MAPE_%": tm.loc[(est, m), "MAPE_%"], "PRUEBA_MAE": tm.loc[(est, m), "MAE"],
            "PRUEBA_WAPE_%": tm.loc[(est, m), "WAPE_%"], "PRUEBA_SESGO_%": tm.loc[(est, m), "Sesgo_%"],
            "PRUEBA_MAPE_BASE_%": tm.loc[(est, BASE_NOMBRE), "MAPE_%"],
            "PRUEBA_MAE_BASE": tm.loc[(est, BASE_NOMBRE), "MAE"],
        })
    sel = pd.DataFrame(filas_sel)
    pred_sel = np.array([pred_test.at[i, seleccion[e][0]] for i, e in pred_test["ESTUDIO"].items()])
    comb = {
        "Selección por modalidad": pred_sel,
        f"Mejor modelo global ({elegido_cv})": pred_test[elegido_cv].to_numpy(),
        "Línea base": pred_test[BASE_NOMBRE].to_numpy(),
    }
    resumen_sel = {}
    for k, v in comb.items():
        r = {"MAPE_medio_%": np.mean([metricas(g["y"], v[g.index])["MAPE_%"]
                                      for _, g in pred_test.groupby("ESTUDIO")]),
             "WAPE_%": metricas(pred_test["y"], v)["WAPE_%"],
             "MAE_log": _mae_log(pred_test["y"], v)}
        resumen_sel[k] = {x: round(y, 3) for x, y in r.items()}
    sel.to_csv(salida / "seleccion_por_modalidad.csv", index=False, float_format="%.4f")
    val_mod.to_csv(salida / "validacion_por_modalidad.csv", index=False, float_format="%.4f")

    tabla.to_csv(salida / "metricas_globales.csv", float_format="%.4f")
    tabla_mod.to_csv(salida / "metricas_por_modalidad.csv", index=False, float_format="%.4f")
    pred_test.to_csv(salida / "predicciones_prueba.csv", index=False, float_format="%.2f")
    oof.loc[val_idx].assign(FECHA=train.loc[val_idx, "FECHA"], ESTUDIO=train.loc[val_idx, "ESTUDIO"]) \
        .to_csv(salida / "predicciones_validacion.csv", index=False, float_format="%.2f")
    # ---- 7. Pronóstico: cada modelo, con sus mejores parámetros, se reentrena con TODOS
    #         los datos (entrenamiento + prueba) y predice los próximos meses ----
    completo = df.copy()
    marcados_todo = []
    if a.tratar_interrupciones:
        completo, marcados_todo = tratar_interrupciones(completo)
    ult = completo["FECHA"].max()
    fechas_f = pd.date_range(ult + pd.offsets.MonthBegin(1), periods=a.horizonte, freq="MS")
    t_ult = completo["T"].max()
    futuro = pd.DataFrame([{"ESTUDIO": e, "FECHA": f, "ANO": f.year, "MES_NUM": f.month,
                            "T": t_ult + i + 1}
                           for e in sorted(completo["ESTUDIO"].unique())
                           for i, f in enumerate(fechas_f)])
    pron = futuro[["ESTUDIO", "FECHA"]].copy()
    pron["Ingenuo estacional"] = ingenuo_estacional_global(completo, futuro)
    pron["Holt-Winters"] = holt_winters_predecir(completo, futuro, tend_ok)
    for nombre, est in estimadores.items():
        m = clone(est).fit(completo[COLS_X], completo["CANTIDAD"])
        pron[nombre] = np.clip(m.predict(futuro[COLS_X]), 0, None)
    for nombre, (cols_e, w) in ens.items():
        pron[nombre] = sum(pron[c] * w[c] for c in cols_e)
    pron.to_csv(salida / "pronostico.csv", index=False, float_format="%.2f")
    print(f"\nPronóstico guardado: {fechas_f[0]:%b %Y} – {fechas_f[-1]:%b %Y} "
          f"({a.horizonte} meses, entrenado con {completo['FECHA'].nunique()} meses)")

    resumen = {"anio_prueba": a.anio_prueba, "modelo_elegido_por_validacion": elegido_cv,
               "datos": Path(a.datos).name,
               "entrenamiento": f"{train['FECHA'].min():%Y-%m} a {train['FECHA'].max():%Y-%m}",
               "prueba": f"{test['FECHA'].min():%Y-%m} a {test['FECHA'].max():%Y-%m}",
               "validacion": f"{len(splits)} bloques de {a.meses_bloque} meses (ventana expansiva)",
               "interrupciones_tratadas": marcados_todo if a.tratar_interrupciones else [],
               "seleccion_por_modalidad": {e: {"modelo": m, "regla": r} for e, (m, r) in seleccion.items()},
               "prueba_seleccion_vs_global": resumen_sel,
               "mejores_parametros": mejores, "segundos_busqueda": {k: round(v, 1) for k, v in tiempos.items()}}
    (salida / "mejores_parametros.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str))

    pd.set_option("display.width", 220)
    cols = ["MASE", "MAPE_%", "WAPE_%", "MAE", "RMSE", "Sesgo_%", "R2", "R2_medio_por_modalidad",
            "CV_MAE_log", "Dif_vs_base_%", "IC95_inf_%", "IC95_sup_%"]
    print(f"\n=== Prueba final en {a.anio_prueba} (ordenado por MASE; <1 = mejor que la línea base) ===")
    print(tabla[cols].round(3).to_string())
    print(f"\nModelo que la validación habría elegido de antemano: {elegido_cv}")
    print("\n=== Selección automática por modalidad (elegida con validación, probada en "
          f"{a.anio_prueba}) ===")
    print(sel[["ESTUDIO", "MODELO", "VAL_MAPE_%", "VAL_MAPE_BASE_%", "PRUEBA_MAPE_%",
               "PRUEBA_MAPE_BASE_%"]].round(1).to_string(index=False))
    for k, v in resumen_sel.items():
        print(f"  {k:45s} MAPE medio {v['MAPE_medio_%']:.1f}%   WAPE {v['WAPE_%']:.1f}%   "
              f"MAE(log) {v['MAE_log']:.4f}")
    print(f"Resultados guardados en: {salida}/")


def ingenuo_estacional_global(hist, futuro):
    pred = pd.Series(index=futuro.index, dtype=float)
    for est, fut in futuro.groupby("ESTUDIO"):
        pred.loc[fut.index] = ingenuo_estacional(hist[hist["ESTUDIO"] == est], fut)
    return pred.to_numpy()


if __name__ == "__main__":
    main()
