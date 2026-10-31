"""
preparar_cemadoja.py
Limpia el Excel de datos abiertos de CEMADOJA (datos.gob.do) y lo deja en el
mismo formato largo que usa la app y la evaluación:
    ESTUDIO, ANO, MES_NUM, MES, CANTIDAD

Fuente: "Estudios Realizados Imágenes, CEMADOJA, 2019 - 2026"
https://datos.gob.do/dataset/0ffff5bf-f3a4-4f23-9706-c12163c72b16

Decisiones de limpieza (todas documentadas aquí):
  1. Se quedan solo estudios de imágenes con serie completa 2019–2026.
     Se excluyen paneles de laboratorio, ecocardiograma, anestesia, y las
     biopsias y drenajes, que cambian de nombre o desaparecen durante el periodo.
  2. Se unifican nombres ("Rayo x" -> "Rayos X", "magentica" -> "magnética", etc.)
     y meses (mayúsculas/minúsculas).
  3. Errores de captura corregidos, verificados por la posición de la fila:
     - Rayos X, febrero, año "202" -> 2024 (está entre las filas de feb-2024).
     - Urografía excretora, "enero 2023" repetida en el bloque de enero 2024 -> 2024.
  4. Picos imposibles de un solo mes (más de 5 veces lo normal) se tratan como
     errores de digitación. Las caídas se conservan.

Uso:
    python preparar_cemadoja.py --entrada cemadoja.xlsx --salida estudios_cemadoja.csv
"""
import argparse
import unicodedata

import pandas as pd

MESES = ["ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO", "JULIO",
         "AGOSTO", "SEPTIEMB", "OCTUBRE", "NOVIEMB", "DICIEMB"]
NOMBRE_MES = {m: i + 1 for i, m in enumerate(
    "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split())}

MODALIDADES = {
    "rayos x convencional": "Rayos X",
    "rayo x convencional": "Rayos X",
    "tomografia": "Tomografías",
    "sonografia": "Sonografías",
    "mamografia": "Mamografías",
    "resonancia magentica": "Resonancias",
    "resonancia magnetica": "Resonancias",
    "densitometria osea": "Densitometrías",
    "fluroscopia": "Fluoroscopias",
    "doppler": "Doppler",
    "urografia excretora": "Urografías excretoras",
}


def normalizar(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--salida", default="estudios_cemadoja.csv")
    a = ap.parse_args()

    df = pd.read_excel(a.entrada)
    df.columns = ["ESTUDIO_ORIG", "CANTIDAD", "MES_ORIG", "ANO"]
    df["clave"] = df["ESTUDIO_ORIG"].map(normalizar)
    df["MES_NUM"] = df["MES_ORIG"].map(normalizar).map(NOMBRE_MES)

    # --- 3. errores de captura, identificados por el contexto de la fila ---
    correcciones = []
    m = (df["clave"].str.startswith("rayo")) & (df["ANO"] == 202) & (df["MES_NUM"] == 2)
    df.loc[m, "ANO"] = 2024
    correcciones.append(f"Rayos X feb, año 202 -> 2024: {m.sum()} fila(s)")

    uro = df[(df["clave"] == "urografia excretora") & (df["ANO"] == 2023) & (df["MES_NUM"] == 1)]
    if len(uro) == 2:
        fila = uro.index.max()  # la segunda aparece dentro del bloque de enero 2024
        vecinos = df.loc[fila - 3:fila + 3, "ANO"].drop(fila)
        if (vecinos == 2024).all():
            df.loc[fila, "ANO"] = 2024
            correcciones.append("Urografía ene-2023 repetida en bloque de ene-2024 -> 2024: 1 fila")

    # --- 1 y 2. solo imágenes con serie completa, nombres unificados ---
    df = df[df["clave"].isin(MODALIDADES)].copy()
    df["ESTUDIO"] = df["clave"].map(MODALIDADES)
    df["CANTIDAD"] = pd.to_numeric(df["CANTIDAD"], errors="raise").astype(int)

    dup = df.duplicated(["ESTUDIO", "ANO", "MES_NUM"], keep=False)
    if dup.any():
        raise SystemExit(f"Quedan duplicados sin resolver:\n{df[dup]}")

    # --- 4. picos imposibles de un solo mes (errores de digitación) ---
    # Regla: el valor supera 5 veces la mediana de su ventana de 13 meses y sus dos
    # vecinos están en niveles normales. Se reemplaza por el promedio de los vecinos.
    # Las CAÍDAS no se tocan: COVID o equipos parados son eventos reales del servicio.
    df = df.sort_values(["ESTUDIO", "ANO", "MES_NUM"]).reset_index(drop=True)
    for est, g in df.groupby("ESTUDIO"):
        s = g["CANTIDAD"].astype(float)
        mediana = s.rolling(13, center=True, min_periods=6).median()
        for i in range(1, len(s) - 1):
            v, ant, sig, med = s.iloc[i], s.iloc[i - 1], s.iloc[i + 1], mediana.iloc[i]
            if med > 0 and v > 5 * med and ant < 2 * med and sig < 2 * med:
                fila = g.index[i]
                nuevo = int(round((ant + sig) / 2))
                correcciones.append(f"{est} {df.at[fila, 'ANO']}-{df.at[fila, 'MES_NUM']:02d}: "
                                    f"pico de {int(v)} (mediana {med:.0f}) -> {nuevo}")
                df.at[fila, "CANTIDAD"] = nuevo

    df["MES"] = df["MES_NUM"].map(lambda n: MESES[n - 1])
    df = df.sort_values(["ESTUDIO", "ANO", "MES_NUM"])[["ESTUDIO", "ANO", "MES_NUM", "MES", "CANTIDAD"]]
    df.to_csv(a.salida, index=False)

    cobertura = df.groupby("ESTUDIO").size()
    print("\n".join(correcciones))
    print(f"Listo: {len(df)} filas, {df['ESTUDIO'].nunique()} modalidades, "
          f"{df['ANO'].min()}–{df['ANO'].max()} -> {a.salida}")
    print("Meses por modalidad:", cobertura.to_dict())


if __name__ == "__main__":
    main()
