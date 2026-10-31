# Pronóstico de demanda de estudios de imágenes

Demo en Streamlit que estima cuántos estudios de imágenes se realizarán en los próximos meses,
con datos abiertos del **Centro de Educación Médica de Amistad Dominico Japonesa (CEMADOJA)**.

Eliges un estudio (mamografías, tomografías, rayos X…) y la app responde en una frase cuántos se
esperan, con su rango probable, un plan mes a mes descargable y qué tan confiable fue el método
cuando se puso a prueba.

## Cómo funciona

- **Datos:** 93 meses de estudios realizados, de enero de 2019 a septiembre de 2026.
- **Nueve métodos en competencia:** desde repetir el mismo mes del año anterior hasta Random
  Forest, Gradient Boosting, SVR, Holt-Winters, regresión Ridge y combinaciones (ensembles).
  Los hiperparámetros se buscaron con GridSearchCV y RandomizedSearchCV.
- **Elección automática por estudio:** se usa el método que menos se equivocó en 2023, 2024 y 2025
  (validación cruzada temporal, aprendiendo siempre solo del pasado). Si ninguno supera en al
  menos un 5% a repetir el año anterior, se usa esa regla.
- **Prueba final a ciegas:** enero a septiembre de 2026, meses que ningún método vio al entrenar.
- **Interrupciones del servicio** (pandemia, equipos parados): se detectan comparando cada mes
  con los 12 anteriores, no se usan para aprender y no cuentan al calificar la validación.

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Reproducir el análisis

```bash
pip install -r requirements-analisis.txt
python analisis/preparar_cemadoja.py --entrada datos_originales/cemadoja_estudios_imagenes_2019_2026.xlsx --salida estudios_cemadoja.csv
python analisis/evaluacion_modelos.py --tratar-interrupciones
python analisis/graficos_evaluacion.py resultados
```

`preparar_cemadoja.py` documenta cada corrección hecha al Excel original: nombres y meses
unificados, dos errores de captura y dos picos imposibles de un solo mes.

## Datos y licencia

- **Fuente:** [Estudios Realizados Imágenes, CEMADOJA, 2019–2026](https://datos.gob.do/dataset/0ffff5bf-f3a4-4f23-9706-c12163c72b16),
  Portal de Datos Abiertos de la República Dominicana (datos.gob.do).
- Los datos originales se publican bajo la licencia
  [Open Data Commons Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
- `estudios_cemadoja.csv` y los archivos de `resultados/` son una base de datos derivada de esa
  fuente y se comparten bajo la misma licencia ODbL 1.0.

## Aviso

Demo educativo elaborado por Alonzo Encarnación. No sustituye la planificación del servicio ni
es una herramienta validada para uso operativo.
