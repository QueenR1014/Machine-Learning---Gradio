# Mantenimiento predictivo de equipos

**Aprendizaje Automático de Máquinas**
Luz Marina Cortés Rico · Andrés Jesús Ospino Stevenson · Juan José Reina Reyes

Aplicación web en Gradio que predice:

- si un equipo va a fallar (clasificación)
- cuánta energía consume (regresión)

Usa seis variables: temperatura, presión, vibración, horas de operación, carga y antigüedad.

## Requisitos

- Python 3.11 (la versión con la que se probó el proyecto).
- Las dependencias de `requirements.txt`.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En macOS/Linux el entorno se activa con `source .venv/bin/activate`.

## Ejecutar la app

```powershell
python app.py
```

Luego abre <http://127.0.0.1:7860>. La primera carga tarda unos segundos porque importa TensorFlow.

La app **no necesita entrenar nada**. Usa los modelos ya entrenados que están en `resultados/`. Si esa carpeta falta o está incompleta, la app la repara así al iniciar:

1. Restaura los modelos optimizados desde `respaldo/optimizados/`.
2. Si no existen, usa el último estado guardado en `respaldo/ultimo/`.
3. Si tampoco hay respaldo, entrena todos los modelos con `optimize=False`, lo que toma unos 20 minutos.

Cada vez que arranca, la app reinicia `models/model_cfg.json` con los valores por defecto de `config/model_options.json`.

## Módulos

| Módulo | Qué hace |
| --- | --- |
| 1 – Exploración | Primeras filas, estadísticas descriptivas, histograma/boxplot de una variable, correlaciones y balance de clases. |
| 2 – Clasificación | Compara las métricas de regresión logística, Random Forest y la red neuronal, con sus matrices de confusión. Predice la falla de un registro nuevo o de un archivo CSV/Excel. |
| 3 – Regresión | Compara las métricas de regresión lineal, SVR y la red neuronal. Predice el consumo de un registro nuevo o de un archivo. |
| 4 – Interactivo | Eliges un modelo, el % de entrenamiento y sus hiperparámetros, y lo reentrenas. Las métricas se comparan con las del modelo pre-entrenado. |

Los módulos 2 y 3 usan los modelos pre-entrenados. El módulo 4 entrena siempre con `optimize=False` y no sobrescribe los modelos guardados: sus resultados viven solo en la sesión.

## Volver a entrenar los modelos

Todos los modelos se entrenan con `entrenar.py`. Sin argumentos muestra un menú:

```powershell
python entrenar.py
```

O directamente con opciones:

| Comando | Qué hace | Tiempo |
| --- | --- | --- |
| `python entrenar.py --rapido` | `optimize=False`: hiperparámetros por defecto de `config/model_options.json` | ~20 minutos |
| `python entrenar.py --optimizar` | `optimize=True`: búsqueda de hiperparámetros (GridSearchCV) | ~2 horas |
| `python entrenar.py --restaurar` | Vuelve a poner en `resultados/` los modelos optimizados del respaldo | segundos |
| `python entrenar.py --restaurar ultimo` | Deshace el último entrenamiento | segundos |
| `python entrenar.py --estado` | Muestra qué hay en `resultados/` y en cada respaldo | — |

Opciones adicionales para `--rapido` y `--optimizar`:

- `--desde-crudos`: limpia de nuevo `data/datos_crudos.csv` y sobrescribe `data/datos_limpios.csv` antes de entrenar.
- `--test-size 0.2`: proporción del conjunto de prueba (por defecto 0.2).
- `--respaldar`: al terminar, guarda el resultado como el nuevo `respaldo/optimizados/`.

Después de entrenar, reinicia la app para que cargue los modelos nuevos.

### Respaldos

Entrenar sobrescribe `resultados/`. Para no perder los modelos optimizados (que tardan cerca de 2 horas):

- **`respaldo/optimizados/`** es la copia protegida de los modelos entrenados con `optimize=True`. Entrenar nunca la toca; solo se reemplaza a propósito, con `--respaldar` o desde el menú.
- **`respaldo/ultimo/`** es una copia automática de `resultados/` justo antes de cada entrenamiento. Sirve para deshacer un entrenamiento hecho por error.

## Estructura

```text
app.py                  punto de entrada: construye y lanza la app
entrenar.py             entrena todos los modelos / maneja respaldos
app/                    interfaz (un archivo por módulo, estilos, gráficas, carga de recursos)
config/
  model_options.json    todos los hiperparámetros modificables, con su valor por defecto y rango
  configManager.py      lee las opciones y escribe models/model_cfg.json
models/                 modelos de clasificación y regresión, TrainingData, model_cfg.json
processing/cleaning.py  limpieza de datos y preparación de registros nuevos
training/
  training.py           entrenamiento, exportación y carga de resultados
  respaldo.py           copia y restauración de resultados/
data/                   datos crudos y limpios
resultados/             modelos, métricas y gráficas que usa la app
respaldo/               copias de resultados/ (ver arriba)
```
