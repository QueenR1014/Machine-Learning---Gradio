import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataManagement import DESCRIPTIVE_COLS as DESCRIPTIVE
from models.classificationModels import NOMBRES as NOMBRES_CLF, classification_metrics
from models.regressionModels import NOMBRES as NOMBRES_REG

ART = Path(__file__).parent / "artifacts"
if not (ART / "modelos.joblib").is_file():
    raise SystemExit("No hay modelos entrenados. Corre primero: python ModelTraining.py")

MODELOS = joblib.load(ART / "modelos.joblib")
RESUMEN = json.loads((ART / "resumen.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(ART / "datos_limpios.csv")
TABLAS = {n: pd.read_csv(ART / f"{n}.csv") for n in [
    "primeras_filas", "descriptivas", "metricas_clasificacion", "umbrales", "metricas_regresion",
    "coeficientes", "parametros_clasificacion", "parametros_regresion"]}
RANGOS = pd.DataFrame(RESUMEN["rangos"])

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan")
CSS = """
.gradio-container table thead th, .gradio-container table thead th * {
    background: #4f46e5 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}
.gradio-container table tbody tr:nth-child(even) td { background-color: #eef2ff; }
.gradio-container table { border-radius: 10px; overflow: hidden; }
"""


def colorear(df, cmap="Blues", axis=0, vmin=None, vmax=None, excluir=(), solo=None, decimales=3):
    num = solo or [c for c in df.select_dtypes("number").columns if c not in excluir]
    return (df.style
            .background_gradient(cmap=cmap, subset=num, axis=axis, vmin=vmin, vmax=vmax, high=0.45, text_color_threshold=0)
            .format({c: f"{{:.{decimales}f}}" if df[c].dtype.kind == "f" else "{}" for c in df.select_dtypes("number").columns}))


def color_falla(v):
    if v == "Sí":
        return "background-color: #fecaca; color: #991b1b; font-weight: 700"
    if v == "No":
        return "background-color: #bbf7d0; color: #166534; font-weight: 700"
    return ""


def graficar(variable, tipo):
    fig, ax = plt.subplots(figsize=(7, 4))
    if tipo == "Histograma":
        sns.histplot(DATA[variable], bins=30, kde=True, color="#4f46e5", edgecolor="black", ax=ax)
        ax.set_ylabel("Frecuencia")
    else:
        sns.boxplot(y=DATA[variable], color="#22d3ee", ax=ax)
    ax.set_title(f"{tipo} de {variable}")
    plt.tight_layout()
    return fig


def registro(*valores):
    if any(v is None for v in valores):
        raise gr.Error("Completa los 6 campos antes de predecir.")
    fuera = [c for c, v in zip(DESCRIPTIVE, valores) if not RANGOS.loc["min", c] <= v <= RANGOS.loc["max", c]]
    if fuera:
        gr.Warning("Fuera del rango de los datos de entrenamiento: " + ", ".join(fuera) +
                   ". La predicción puede no ser confiable.")
    return pd.DataFrame([valores], columns=DESCRIPTIVE)


def escalar(scaler, x):
    return pd.DataFrame(scaler.transform(x), columns=DESCRIPTIVE)


def predecir_falla(*args):
    *valores, umbral = args
    x = escalar(MODELOS["clf_scaler"], registro(*valores))
    filas = []
    for key, m in MODELOS["clf_models"].items():
        prob = m.predict_proba(x)[0, 1]
        filas.append([NOMBRES_CLF[key], "Sí" if prob >= umbral else "No", round(prob, 3)])
    df = pd.DataFrame(filas, columns=["Modelo", "Falla", "Prob. falla"])
    return colorear(df, "Reds", vmin=0, vmax=1).map(color_falla, subset=["Falla"])


def predecir_consumo(*valores):
    x = escalar(MODELOS["reg_scaler"], registro(*valores))
    filas = [[NOMBRES_REG[key], round(float(m.predict(x)[0]), 2)] for key, m in MODELOS["reg_models"].items()]
    return colorear(pd.DataFrame(filas, columns=["Modelo", "Consumo (kWh)"]), "Blues")


def entrenar_rf(pct, n_arboles, profundidad):
    X, y = DATA[DESCRIPTIVE], DATA["falla_del_equipo"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=pct / 100, random_state=2026, stratify=y)
    esc = StandardScaler().fit(Xtr)
    rf = RandomForestClassifier(n_estimators=int(n_arboles), max_depth=int(profundidad), random_state=2026, n_jobs=-1)
    rf.fit(esc.transform(Xtr), ytr)
    pred = rf.predict(esc.transform(Xte))

    met = pd.DataFrame([classification_metrics(yte, pred)]).round(3)
    met.insert(0, "Train / Test", f"{len(Xtr)} / {len(Xte)}")

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix(yte, pred, labels=[0, 1]),
                           display_labels=["No", "Sí"]).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title("Random Forest reentrenado")
    plt.tight_layout()
    return colorear(met, "Greens", axis=None), fig


def entradas():
    with gr.Row():
        t = gr.Number(55, label="Temperatura (°C)")
        p = gr.Number(5, label="Presión (bar)")
        v = gr.Number(2.5, label="Vibración (mm/s)")
    with gr.Row():
        h = gr.Number(10000, label="Horas de operación")
        c = gr.Number(45, label="Carga (%)")
        a = gr.Number(5, label="Antigüedad (años)")
    return [t, p, v, h, c, a]


with gr.Blocks(title="Caso Final ML") as app:
    gr.Markdown("# Desempeño energético y fallas en equipos industriales")

    with gr.Tab("1. Exploración"):
        gr.Markdown(f"Filas usadas tras la limpieza: **{RESUMEN['filas_usadas']}** de {RESUMEN['filas_originales']}")
        gr.Dataframe(colorear(TABLAS["primeras_filas"], "Blues"), label="Primeras filas")
        gr.Dataframe(colorear(TABLAS["descriptivas"], "Purples", axis=1, excluir=("count",)), label="Estadísticas descriptivas")
        with gr.Row():
            var = gr.Dropdown(list(DATA.columns), value="temperatura", label="Variable")
            tipo = gr.Radio(["Histograma", "Boxplot"], value="Histograma", label="Tipo de gráfico")
        btn1 = gr.Button("Graficar", variant="primary")
        graf = gr.Plot()
        btn1.click(graficar, [var, tipo], graf)
        gr.Markdown("### Mapa de correlaciones")
        gr.Image(str(ART / "correlaciones.png"), show_label=False)

    with gr.Tab("2. Clasificación"):
        gr.Markdown(f"Mejor modelo (por F1): **{RESUMEN['mejor_clasificacion']}**")
        gr.Dataframe(colorear(TABLAS["metricas_clasificacion"], "Greens"), label="Métricas (conjunto de prueba) + F1 por validación cruzada")
        gr.Markdown("### Matrices de confusión")
        gr.Image(str(ART / "matrices_confusion.png"), show_label=False)
        with gr.Accordion("Hiperparámetros elegidos (GridSearch) y umbral de decisión", open=False):
            gr.Dataframe(TABLAS["parametros_clasificacion"], label="Hiperparámetros")
            gr.Dataframe(colorear(TABLAS["umbrales"], "Oranges"),
                         label=f"{RESUMEN['mejor_clasificacion']} según el umbral")
        gr.Markdown("### Nuevo registro")
        ent = entradas()
        umbral = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Umbral de decisión (prob. mínima para decir 'Sí')")
        btn2 = gr.Button("Predecir falla", variant="primary")
        sal = gr.Dataframe(headers=["Modelo", "Falla", "Prob. falla"], label="Predicciones")
        btn2.click(predecir_falla, ent + [umbral], sal)

    with gr.Tab("3. Regresión"):
        gr.Markdown(f"Mejor modelo (por R²): **{RESUMEN['mejor_regresion']}**")
        gr.Dataframe(colorear(TABLAS["metricas_regresion"], "Greens", solo=["R2"], decimales=4), label="Métricas (conjunto de prueba)")
        gr.Markdown("### Consumo real vs predicho")
        gr.Image(str(ART / "real_vs_predicho.png"), show_label=False)
        with gr.Accordion("Hiperparámetros elegidos (GridSearch) y peso de las variables", open=False):
            gr.Dataframe(TABLAS["parametros_regresion"], label="Hiperparámetros")
            gr.Image(str(ART / "coeficientes.png"), show_label=False)
        gr.Markdown("### Nuevo registro")
        ent2 = entradas()
        btn3 = gr.Button("Predecir consumo", variant="primary")
        sal2 = gr.Dataframe(headers=["Modelo", "Consumo (kWh)"], label="Predicciones")
        btn3.click(predecir_consumo, ent2, sal2)

    with gr.Tab("4. Interactivo"):
        gr.Markdown("Random Forest para clasificar la falla: cambia los parámetros y reentrena.")
        pct = gr.Slider(50, 90, value=80, step=5, label="% de entrenamiento")
        arb = gr.Slider(10, 500, value=100, step=10, label="Número de árboles (n_estimators)")
        prof = gr.Slider(2, 30, value=10, step=1, label="Profundidad máxima (max_depth)")
        btn4 = gr.Button("Entrenar", variant="primary")
        met = gr.Dataframe(headers=["Train / Test", "Accuracy", "Recall", "Specificity", "Precision", "F1"], label="Métricas")
        cm_plot = gr.Plot()
        btn4.click(entrenar_rf, [pct, arb, prof], [met, cm_plot])


def lanzar(share=False):
    app.launch(share=share, theme=THEME, css=CSS)


if __name__ == "__main__":
    lanzar()
