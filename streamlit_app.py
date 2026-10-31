"""
streamlit_app.py — Pronóstico de demanda de estudios de imágenes (CEMADOJA, datos abiertos).

Versión para usuario final. Al elegir un estudio, la app usa automáticamente el método
que mejor predijo ese estudio en las pruebas 2023–2025 (elegido sin mirar 2026) y muestra
cómo le fue en 2026, meses que ningún método había visto.

La app no entrena: lee lo que genera `analisis/evaluacion_modelos.py` en resultados/.

Ejecutar desde esta carpeta:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

AQUI = Path(__file__).parent
RES = AQUI / "resultados" if (AQUI / "resultados").exists() else AQUI
BASE = "Ingenuo estacional"

CLARO = {"tinta": "#14212B", "tinta2": "#4E5D68", "linea": "#D6DEE3", "acento": "#0F7C8C",
         "fondo": "#F5F7F8", "superficie": "#E6EDF0", "naranja": "#B4561F"}
OSCURO = {"tinta": "#E6EDF1", "tinta2": "#A3B1BA", "linea": "#2A3842", "acento": "#3FB8C6",
          "fondo": "#0F171D", "superficie": "#1B2830", "naranja": "#E08A55"}
FUENTE_TXT = "Atkinson Hyperlegible"

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]

# Cómo se nombra cada estudio dentro de una frase ("esperamos unas 270 mamografías")
EN_FRASE = {
    "Densitometrías": ("unas", "densitometrías"), "Doppler": ("unos", "estudios Doppler"),
    "Fluoroscopias": ("unas", "fluoroscopias"), "Mamografías": ("unas", "mamografías"),
    "Rayos X": ("unos", "estudios de rayos X"), "Resonancias": ("unas", "resonancias"),
    "Sonografías": ("unas", "sonografías"), "Tomografías": ("unas", "tomografías"),
    "Urografías excretoras": ("unas", "urografías excretoras"),
}

# Cada método explicado en lenguaje simple
METODO = {
    "Ingenuo estacional": "repetir lo que se hizo el mismo mes del año anterior",
    "Holt-Winters": "un método estadístico que sigue el nivel de cada mes y su patrón a lo largo del año",
    "Ridge": "una regresión que combina la tendencia con el efecto de cada mes",
    "Random Forest": "un modelo de IA con cientos de árboles de decisión que votan la respuesta",
    "Gradient Boosting": "un modelo de IA con árboles de decisión que se corrigen unos a otros",
    "SVR": "un modelo de IA que busca la curva que mejor se ajusta a la historia",
    "Ensemble promedio top 3": "la combinación de tres modelos de IA, que promedia sus predicciones",
    "Ensemble ponderado top 3": "la combinación de tres modelos de IA, dando más peso al más preciso",
}
NOMBRE = {
    "Ingenuo estacional": "Repetir el año anterior", "Holt-Winters": "Holt-Winters",
    "Ridge": "Regresión Ridge", "Random Forest": "Random Forest", "Gradient Boosting": "Gradient Boosting",
    "SVR": "SVR", "Ensemble promedio top 3": "Combinación de 3 modelos",
    "Ensemble ponderado top 3": "Combinación ponderada de 3 modelos",
}


def metodo(m):
    if m.startswith("Ensemble promedio ("):
        return "la combinación de dos modelos de IA, que promedia sus predicciones"
    return METODO.get(m, m)


def nombre(m):
    if m.startswith("Ensemble promedio ("):
        return "Combinación de 2 modelos"
    return NOMBRE.get(m, m)


def mes_largo(f):
    return f"{MESES[f.month - 1]} de {f.year}"


def mes_corto(f):
    return f"{MESES[f.month - 1][:3]} {f.year}"


def n(x):
    return f"{x:,.0f}"


# =============================================================================
# Datos
# =============================================================================
@st.cache_data
def cargar():
    real = pd.read_csv(AQUI / "estudios_cemadoja.csv")
    real["FECHA"] = pd.to_datetime(dict(year=real["ANO"], month=real["MES_NUM"], day=1))
    pron = pd.read_csv(RES / "pronostico.csv", parse_dates=["FECHA"])
    prueba = pd.read_csv(RES / "predicciones_prueba.csv", parse_dates=["FECHA"])
    sel = pd.read_csv(RES / "seleccion_por_modalidad.csv").set_index("ESTUDIO")
    val = pd.read_csv(RES / "validacion_por_modalidad.csv")
    pru = pd.read_csv(RES / "metricas_por_modalidad.csv")
    info = json.loads((RES / "mejores_parametros.json").read_text(encoding="utf-8"))
    cortes = {tuple(x.rsplit(" ", 1)) for x in info.get("interrupciones_tratadas", [])}
    real["INTERRUPCION"] = [(e, f"{f:%Y-%m}") in cortes for e, f in zip(real["ESTUDIO"], real["FECHA"])]
    return real, pron, prueba, sel, val, pru, info


st.set_page_config(page_title="Pronóstico de demanda de imágenes", page_icon="🩻", layout="wide")
real, pron, prueba, sel, val, pru, info = cargar()

# Encabezado: título grande a la izquierda, modo oscuro a la derecha
cab, interruptor = st.columns([6, 1])
oscuro = interruptor.toggle("Modo oscuro", value=False)
P = OSCURO if oscuro else CLARO
TINTA, TINTA_2, LINEA, ACENTO, FONDO = P["tinta"], P["tinta2"], P["linea"], P["acento"], P["fondo"]
SUPERFICIE, NARANJA = P["superficie"], P["naranja"]

# =============================================================================
# Estilo
# =============================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, label, table, [data-testid="stMetric"] {{
  font-family: '{FUENTE_TXT}', system-ui, sans-serif !important;
}}
.block-container {{ max-width: 1180px; padding-top: 4.5rem !important; }}
#MainMenu, footer {{ visibility: hidden; }}
.stApp, [data-testid="stAppViewContainer"], header[data-testid="stHeader"] {{ background: {FONDO} !important; }}
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
.stRadio label p {{ color: {TINTA} !important; }}
.stApp p.titulo {{ font-size: clamp(2.1rem, 3.8vw, 2.9rem) !important; line-height: 1.1 !important;
  font-weight: 700 !important; color: {TINTA} !important; letter-spacing: -0.015em; margin: 0 0 .5rem 0 !important; }}
.stApp p.subtitulo {{ font-size: 1.05rem !important; color: {TINTA_2} !important; margin: 0 0 1.2rem 0 !important;
  max-width: 75ch; }}
.stApp p.respuesta {{ font-size: clamp(1.6rem, 2.8vw, 2.2rem) !important; line-height: 1.18 !important;
  font-weight: 700 !important; color: {TINTA} !important; margin: 1.4rem 0 .5rem 0 !important; max-width: 34ch; }}
.stApp p.detalle {{ font-size: 1.15rem !important; line-height: 1.55 !important; color: {TINTA_2} !important;
  max-width: 62ch; margin: 0 0 .4rem 0 !important; }}
.stApp p.detalle b {{ color: {TINTA}; }}
.stApp p.seccion {{ font-size: 1.3rem !important; font-weight: 700 !important; color: {TINTA} !important;
  margin: 2.2rem 0 .3rem 0 !important; }}
.stApp p.nota {{ font-size: .95rem !important; line-height: 1.55 !important; color: {TINTA_2} !important; max-width: 70ch; }}
.confianza {{ border-left: 4px solid {ACENTO}; padding: .2rem 0 .2rem 1rem; margin: .4rem 0 1rem 0; }}
.confianza.cautela {{ border-left-color: {NARANJA}; }}
.stApp .confianza p {{ margin: .25rem 0 !important; font-size: 1.02rem !important; line-height: 1.55 !important;
  color: {TINTA} !important; max-width: 62ch; }}
.etiqueta {{ font-weight: 700; }}
.stApp p.pie {{ font-size: .85rem !important; color: {TINTA_2} !important; border-top: 1px solid {LINEA};
  padding-top: .8rem; margin-top: 2.5rem !important; }}
/* Tablas */
[data-testid="stTable"] table {{ width: 100%; border-collapse: collapse; font-size: .95rem; }}
[data-testid="stTable"] th, [data-testid="stTable"] td {{ color: {TINTA} !important; background: {FONDO} !important;
  border: none !important; border-bottom: 1px solid {LINEA} !important; padding: .45rem .6rem !important; }}
[data-testid="stTable"] thead th {{ color: {TINTA_2} !important; font-weight: 700 !important; }}
/* Controles */
div[data-baseweb="select"] > div {{ background-color: {SUPERFICIE} !important; border-color: {LINEA} !important; }}
div[data-baseweb="select"] * {{ color: {TINTA} !important; fill: {TINTA} !important; }}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li {{ background: {SUPERFICIE} !important; color: {TINTA} !important; }}
[data-testid="stExpander"] details {{ background: {FONDO} !important; border-color: {LINEA} !important; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: {TINTA} !important; fill: {TINTA} !important; }}
[data-testid="stDownloadButton"] button {{ background: {SUPERFICIE} !important; color: {TINTA} !important;
  border-color: {LINEA} !important; }}
[data-testid="stDownloadButton"] button p {{ color: {TINTA} !important; }}
</style>
""", unsafe_allow_html=True)


def grafico_base(chart):
    return chart.configure(font=FUENTE_TXT, background=FONDO) \
        .configure_view(strokeWidth=0) \
        .configure_axis(labelColor=TINTA_2, titleColor=TINTA_2, gridColor=LINEA, domain=False,
                        tickColor=LINEA, labelFontSize=12, titleFontSize=12) \
        .configure_legend(labelColor=TINTA, labelFontSize=12, orient="top", title=None)


# =============================================================================
# Encabezado y controles
# =============================================================================
cab.markdown('<p class="titulo">Pronóstico de demanda de estudios de imágenes</p>'
             '<p class="subtitulo">Centro de Educación Médica de Amistad Dominico Japonesa (CEMADOJA), '
             'con sus datos abiertos de enero de 2019 a septiembre de 2026.</p>', unsafe_allow_html=True)

estudios = sorted(real["ESTUDIO"].unique())
c1, c2 = st.columns([3, 2])
estudio = c1.selectbox("¿Qué estudio quieres planificar?", estudios, index=estudios.index("Mamografías"))
horizonte = c2.radio("¿Cuántos meses hacia adelante?", [3, 6, 12], index=1, horizontal=True,
                     format_func=lambda x: f"{x} meses")

s = sel.loc[estudio]
modelo = s["MODELO"]
hist = real[real["ESTUDIO"] == estudio].sort_values("FECHA")
ult = hist["FECHA"].max()
margen = float(s["PRUEBA_MAE"])

f = pron[pron["ESTUDIO"] == estudio].sort_values("FECHA").head(horizonte)[["FECHA", modelo]] \
    .rename(columns={modelo: "ESPERADO"}).reset_index(drop=True)
f["ESPERADO"] = f["ESPERADO"].round()
f["MIN"] = (f["ESPERADO"] - margen).clip(lower=0).round()
f["MAX"] = (f["ESPERADO"] + margen).round()

# =============================================================================
# La respuesta
# =============================================================================
prox = f.iloc[0]
pico = f.loc[f["ESPERADO"].idxmax()]
articulo, frase = EN_FRASE.get(estudio, ("unos", f"estudios de {estudio.lower()}"))
st.markdown(f'<p class="respuesta">En {mes_largo(prox["FECHA"])} esperamos {articulo} {n(prox["ESPERADO"])} {frase}.</p>',
            unsafe_allow_html=True)
detalle = f'Lo más probable es que sean entre <b>{n(prox["MIN"])} y {n(prox["MAX"])}</b>. '
detalle += f'En los próximos {horizonte} meses serían {articulo} <b>{n(f["ESPERADO"].sum())}</b> en total'
detalle += f', y el mes con más carga sería <b>{mes_largo(pico["FECHA"])}</b>.' if horizonte > 1 else "."
st.markdown(f'<p class="detalle">{detalle}</p>', unsafe_allow_html=True)

# =============================================================================
# Gráfico: lo que pasó y lo que viene
# =============================================================================
h = hist[hist["FECHA"] > ult - pd.DateOffset(months=24)].copy()
h["ESTADO"] = h["INTERRUPCION"].map({True: "Servicio interrumpido", False: "Realizados"})
puente = pd.DataFrame({"FECHA": [ult], "ESPERADO": [h["CANTIDAD"].iloc[-1]]})
x = alt.X("FECHA:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=0, tickCount=8))
y_tit = "Estudios por mes"

banda = alt.Chart(f).mark_area(color=ACENTO, opacity=0.16).encode(x=x, y=alt.Y("MIN:Q", title=y_tit), y2="MAX:Q")
linea_h = alt.Chart(h).mark_line(color=TINTA, strokeWidth=2).encode(x=x, y="CANTIDAD:Q")
puntos_h = alt.Chart(h).mark_point(size=60, filled=True, strokeWidth=2).encode(
    x=x, y="CANTIDAD:Q",
    shape=alt.Shape("ESTADO:N", scale=alt.Scale(domain=["Realizados", "Servicio interrumpido"],
                                                  range=["circle", "diamond"]), legend=None),
    color=alt.Color("ESTADO:N", scale=alt.Scale(domain=["Realizados", "Servicio interrumpido"],
                                                range=[TINTA, NARANJA]), legend=None),
    tooltip=[alt.Tooltip("FECHA:T", title="Mes", format="%B %Y"),
             alt.Tooltip("CANTIDAD:Q", title="Realizados", format=",.0f"),
             alt.Tooltip("ESTADO:N", title="Estado")])
linea_f = alt.Chart(pd.concat([puente, f[["FECHA", "ESPERADO"]]])).mark_line(
    color=ACENTO, strokeWidth=2.5, strokeDash=[7, 4]).encode(x=x, y="ESPERADO:Q")
puntos_f = alt.Chart(f).mark_point(size=70, filled=True, color=ACENTO).encode(
    x=x, y="ESPERADO:Q",
    tooltip=[alt.Tooltip("FECHA:T", title="Mes", format="%B %Y"),
             alt.Tooltip("ESPERADO:Q", title="Esperados", format=",.0f"),
             alt.Tooltip("MIN:Q", title="Desde", format=",.0f"),
             alt.Tooltip("MAX:Q", title="Hasta", format=",.0f")])
hoy = alt.Chart(pd.DataFrame({"FECHA": [ult], "t": [f"Último dato: {mes_corto(ult)}"]}))
regla = hoy.mark_rule(color=TINTA_2, strokeDash=[2, 3]).encode(x="FECHA:T")
texto = hoy.mark_text(align="right", dx=-6, dy=-150, color=TINTA_2, fontSize=12).encode(x="FECHA:T", text="t:N")
st.altair_chart(grafico_base((banda + regla + texto + linea_h + puntos_h + linea_f + puntos_f)
                             .properties(height=340)), use_container_width=True)
leyenda = "Línea oscura: estudios realizados. Línea verde azulada: lo esperado, con su rango probable sombreado."
if h["INTERRUPCION"].any():
    leyenda += " Rombos naranja: meses con el servicio interrumpido, que no se usaron para aprender."
st.markdown(f'<p class="nota">{leyenda}</p>', unsafe_allow_html=True)

# =============================================================================
# Plan mes a mes y confiabilidad
# =============================================================================
izq, der = st.columns([11, 10], gap="large")

with izq:
    st.markdown('<p class="seccion">Mes a mes</p>', unsafe_allow_html=True)
    normales = hist[~hist["INTERRUPCION"]]
    ref = normales[normales["FECHA"] > ult - pd.DateOffset(months=12)]["CANTIDAD"].mean()
    por_fecha = hist.set_index("FECHA")
    filas = []
    for _, r in f.iterrows():
        antes = r["FECHA"] - pd.DateOffset(years=1)
        if antes in por_fecha.index and por_fecha.at[antes, "CANTIDAD"] > 0 and not por_fecha.at[antes, "INTERRUPCION"]:
            vs = f"{(r['ESPERADO'] / por_fecha.at[antes, 'CANTIDAD'] - 1) * 100:+.0f}%"
        else:
            vs = "sin dato comparable"
        rel = r["ESPERADO"] / ref - 1 if ref else 0
        carga = "▲ Alta" if rel > 0.10 else ("▼ Baja" if rel < -0.10 else "● Normal")
        filas.append({"Mes": mes_largo(r["FECHA"]).capitalize(), "Esperados": n(r["ESPERADO"]),
                      "Rango probable": f"{n(r['MIN'])} a {n(r['MAX'])}",
                      "Frente al año anterior": vs, "Carga": carga})
    plan = pd.DataFrame(filas)
    st.table(plan.set_index("Mes"))
    st.markdown(f'<p class="nota">La carga compara cada mes con el promedio de los últimos 12 meses '
                f'({n(ref)} al mes): alta si lo supera en más de un 10%, baja si queda más de un 10% por debajo.</p>',
                unsafe_allow_html=True)
    st.download_button("Descargar el plan", plan.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"plan_{estudio.lower().replace(' ', '_')}.csv", mime="text/csv")

with der:
    st.markdown('<p class="seccion">¿Qué tan confiable es?</p>', unsafe_allow_html=True)
    err, err_base = float(s["PRUEBA_MAPE_%"]), float(s["PRUEBA_MAPE_BASE_%"])
    es_base = modelo == BASE
    nivel = "alta" if err < 15 else ("media" if err < 30 else "baja")
    cautela = nivel == "baja"

    p1 = (f'Lo pusimos a prueba: aprendió con los datos hasta diciembre de 2025 y predijo enero a '
          f'septiembre de 2026 sin haberlos visto. Se desvió en promedio un <b>{err:.0f}%</b> '
          f'(unos {n(margen)} estudios al mes).')
    if es_base:
        p2 = ("Para este estudio, ningún modelo de IA superó con claridad a la regla de repetir el año "
              "anterior en las pruebas de 2023 a 2025, así que usamos esa regla.")
    elif err <= err_base:
        p2 = f'Repetir lo del año anterior se habría desviado un {err_base:.0f}%.'
    elif not cautela:
        p2 = (f'En 2026, repetir lo del año anterior lo habría hecho algo mejor ({err_base:.0f}%). '
              f'Las dos opciones sirven como guía para planificar.')
    else:
        p2 = (f'En 2026, repetir lo del año anterior lo habría hecho mejor ({err_base:.0f}%).')
    if cautela:
        p2 += (" Este estudio varía mucho de un mes a otro, así que conviene usar el pronóstico como "
               "referencia, no como cifra exacta.")
    etiqueta = f'Confiabilidad {nivel}' + (", úsalo con cautela" if cautela else "")
    st.markdown(f'<div class="confianza{" cautela" if cautela else ""}"><p class="etiqueta">{etiqueta}</p>'
                f'<p>{p1}</p><p>{p2}</p></div>', unsafe_allow_html=True)

    t = prueba[prueba["ESTUDIO"] == estudio][["FECHA", "y", modelo]] \
        .rename(columns={"y": "Realizados", modelo: "Lo que predijo"}) \
        .melt("FECHA", var_name="Serie", value_name="CANTIDAD")
    esc = alt.Scale(domain=["Realizados", "Lo que predijo"], range=[TINTA, ACENTO])
    g = alt.Chart(t).encode(
        x=alt.X("FECHA:T", title=None, axis=alt.Axis(format="%b", labelAngle=0)),
        y=alt.Y("CANTIDAD:Q", title="Estudios en 2026"),
        color=alt.Color("Serie:N", scale=esc),
        strokeDash=alt.StrokeDash("Serie:N", scale=alt.Scale(domain=["Realizados", "Lo que predijo"],
                                                             range=[[1, 0], [6, 3]]), legend=None),
        tooltip=[alt.Tooltip("FECHA:T", title="Mes", format="%B %Y"), alt.Tooltip("Serie:N", title=" "),
                 alt.Tooltip("CANTIDAD:Q", title="Estudios", format=",.0f")])
    st.altair_chart(grafico_base((g.mark_line(strokeWidth=2.2) + g.mark_point(size=50, filled=True))
                                 .properties(height=230)), use_container_width=True)
    st.markdown(f'<p class="nota"><b>Método usado:</b> {metodo(modelo)}. Se eligió automáticamente '
                f'porque fue el que menos se equivocó con {frase} en las pruebas de 2023, 2024 y 2025.</p>',
                unsafe_allow_html=True)

# =============================================================================
# Cómo funciona
# =============================================================================
with st.expander("Cómo se hace este pronóstico"):
    st.markdown(
        "- **Los datos:** 93 meses de estudios realizados en CEMADOJA, de enero de 2019 a septiembre "
        "de 2026, publicados como datos abiertos del Estado dominicano.\n"
        "- **Nueve métodos en competencia:** desde repetir el año anterior hasta combinaciones de "
        "modelos de inteligencia artificial.\n"
        "- **Una elección por estudio:** para cada estudio se queda el método que menos se equivocó "
        "en 2023, 2024 y 2025, aprendiendo siempre solo de los años anteriores. Si ninguno supera con "
        "claridad a repetir el año anterior, se usa esa regla.\n"
        "- **Una prueba final a ciegas:** después se comprueba con enero a septiembre de 2026, "
        "meses que ningún método había visto.\n"
        "- **Los meses con el servicio interrumpido** (la pandemia o un equipo parado) no se usan para "
        "aprender, porque nadie puede anticiparlos.")
    v = val[val["ESTUDIO"] == estudio].set_index("Modelo")["VAL_MAPE_%"]
    p = pru[pru["ESTUDIO"] == estudio].set_index("Modelo")["MAPE_%"]
    ranking = pd.DataFrame({"Error en las pruebas 2023–2025": v, "Error en 2026": p}).sort_values(
        "Error en las pruebas 2023–2025")
    ranking.index = [nombre(m) + ("  ✓ elegido" if m == modelo else "") for m in ranking.index]
    ranking.index.name = f"Métodos evaluados para {frase}"
    st.table(ranking.apply(lambda c: c.map(lambda x: f"{x:.0f}%")))
    st.markdown('<p class="nota">El error es la desviación promedio entre lo esperado y lo realizado, '
                'en porcentaje. El orden de la tabla es el de las pruebas 2023–2025, que es lo único '
                'que se usó para elegir.</p>', unsafe_allow_html=True)

st.markdown('<p class="pie">Fuente: Estudios Realizados Imágenes, CEMADOJA, 2019–2026, publicado en '
            'datos.gob.do con licencia ODbL. Demo educativo elaborado por Alonzo Encarnación; no sustituye '
            'la planificación del servicio.</p>', unsafe_allow_html=True)
