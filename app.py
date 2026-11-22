import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from ModelTraining import train_all, build_structure, DESCRIPTIVE
from models.classificationModels import NOMBRES as NOMBRES_CLF, classification_metrics, threshold_table
from models.regressionModels import NOMBRES as NOMBRES_REG

print("Entrenando modelos...")
#CFG, DS_CLF, DS_REG, RES_CLF, RES_REG = train_all()
#DATA = DS_CLF.data
print("Modelos listos.")

RANGOS = DATA[DESCRIPTIVE].agg(["min", "max"])


def graficar(variable, tipo):
    fig, ax = plt.subplots(figsize=(7, 4))
    if tipo == "Histograma":
        ax.hist(DATA[variable], bins=30, color="steelblue", edgecolor="black")
        ax.set_ylabel("Frecuencia")
    else:
        sns.boxplot(y=DATA[variable], ax=ax, color="orange")
    ax.set_title(f"{tipo} de {variable}")
    ax.set_xlabel(variable if tipo == "Histograma" else "")
    plt.tight_layout()
    return fig


def fig_correlaciones():
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(DATA.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Mapa de correlaciones")
    plt.tight_layout()
    return fig


def fig_matrices():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, cm) in zip(axes, RES_CLF["matrices"].items()):
        ConfusionMatrixDisplay(cm, display_labels=["No", "Sí"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(NOMBRES_CLF[key])
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


def predecir_falla(*args):
    *valores, umbral = args
    x = DS_CLF.transform_new(registro(*valores))
    filas = []
    for key, m in RES_CLF["models"].items():
        prob = m.predict_proba(x)[0, 1]
        filas.append([NOMBRES_CLF[key], "Sí" if prob >= umbral else "No", round(prob, 3)])
    return pd.DataFrame(filas, columns=["Modelo", "Falla", "Prob. falla"])


def tabla_umbrales():
    key = [k for k, v in NOMBRES_CLF.items() if v == RES_CLF["best"]][0]
    return threshold_table(RES_CLF["models"][key], DS_CLF.X_test_scaled, DS_CLF.y_test)


def predecir_consumo(*valores):
    x = DS_REG.transform_new(registro(*valores))
    filas = [[NOMBRES_REG[key], round(float(m.predict(x)[0]), 2)] for key, m in RES_REG["models"].items()]
    return pd.DataFrame(filas, columns=["Modelo", "Consumo (kWh)"])


def fig_real_vs_pred():
    key = [k for k, v in NOMBRES_REG.items() if v == RES_REG["best"]][0]
    y, p = DS_REG.y_test, RES_REG["predictions"][key]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y, p, s=8, alpha=0.5, color="purple")
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    ax.set_xlabel("Consumo real"); ax.set_ylabel("Consumo predicho")
    ax.set_title(f"Real vs predicho ({RES_REG['best']})")
    plt.tight_layout()
    return fig


def entrenar_rf(pct, n_arboles, profundidad):
    ds = build_structure("classification", test_size=1 - pct / 100)
    params = dict(CFG["classification"]["RandomForest"])
    params.update(n_estimators=int(n_arboles), max_depth=int(profundidad))
    rf = RandomForestClassifier(**params)
    rf.fit(ds.X_train_scaled, ds.y_train)
    pred = rf.predict(ds.X_test_scaled)

    met = pd.DataFrame([classification_metrics(ds.y_test, pred)]).round(3)
    met.insert(0, "Train / Test", f"{len(ds.X_train)} / {len(ds.X_test)}")

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix(ds.y_test, pred, labels=[0, 1]),
                           display_labels=["No", "Sí"]).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title("Random Forest reentrenado")
    plt.tight_layout()
    return met, fig


def entradas():
    return [gr.Number(55, label="Temperatura (°C)"), gr.Number(5, label="Presión (bar)"),
            gr.Number(2.5, label="Vibración (mm/s)"), gr.Number(20000, label="Horas de operación"),
            gr.Number(50, label="Carga (%)"), gr.Number(10, label="Antigüedad (años)")]


with gr.Blocks(title="Caso Final ML") as app:
    gr.Markdown("# Desempeño energético y fallas en equipos industriales")

    with gr.Tab("1. Exploración"):
        gr.Markdown(f"Filas usadas tras la limpieza: **{len(DATA)}** de {len(DS_CLF.raw_data)}")
        gr.Dataframe(DATA.head(10), label="Primeras filas")
        gr.Dataframe(DATA.describe().T.round(2).reset_index(), label="Estadísticas descriptivas")
        with gr.Row():
            var = gr.Dropdown(list(DATA.columns), value="temperatura", label="Variable")
            tipo = gr.Radio(["Histograma", "Boxplot"], value="Histograma", label="Tipo de gráfico")
        btn = gr.Button("Graficar")
        graf = gr.Plot()
        btn.click(graficar, [var, tipo], graf)
        gr.Plot(fig_correlaciones())

    with gr.Tab("2. Clasificación"):
        gr.Markdown(f"Mejor modelo (por F1 en validación cruzada): **{RES_CLF['best']}**. "
                    "Las diferencias entre modelos son menores que la desviación de la CV: en la práctica empatan.")
        gr.Dataframe(RES_CLF["metrics"], label="Métricas (test) + F1 por validación cruzada (train)")
        gr.Plot(fig_matrices())
        gr.Markdown("### Umbral de decisión\nBajar el umbral detecta más fallas (recall ↑) a costa de más falsas alarmas (precision ↓).")
        gr.Dataframe(tabla_umbrales(), label=f"{RES_CLF['best']} según el umbral")
        gr.Markdown("### Nuevo registro")
        ent = entradas()
        umbral = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Umbral de decisión (prob. mínima para decir 'Sí')")
        btn2 = gr.Button("Predecir falla")
        sal = gr.Dataframe(label="Predicciones")
        btn2.click(predecir_falla, ent + [umbral], sal)

    with gr.Tab("3. Regresión"):
        gr.Markdown(f"Mejor modelo (por R²): **{RES_REG['best']}**")
        gr.Markdown("R² ≈ 1 en todos: el consumo es casi lineal en las variables (sobre todo la carga), por eso "
                    "el modelo más simple basta. La línea base (predecir la media) da R² ≈ 0.")
        gr.Dataframe(pd.concat([RES_REG["metrics"], RES_REG["baseline"]], ignore_index=True), label="Métricas")
        gr.Dataframe(RES_REG["coefs"], label="Peso de cada variable en la regresión lineal (variables estandarizadas)")
        gr.Plot(fig_real_vs_pred())
        gr.Markdown("### Nuevo registro")
        ent2 = entradas()
        btn3 = gr.Button("Predecir consumo")
        sal2 = gr.Dataframe(label="Predicciones")
        btn3.click(predecir_consumo, ent2, sal2)

    with gr.Tab("4. Interactivo"):
        gr.Markdown("Random Forest para clasificar la falla: cambia los parámetros y reentrena.")
        pct = gr.Slider(50, 90, value=round((1 - CFG["test_size"]) * 100), step=5, label="% entrenamiento")
        arb = gr.Slider(10, 500, value=CFG["classification"]["RandomForest"]["n_estimators"], step=10, label="Número de árboles (n_estimators)")
        prof = gr.Slider(2, 30, value=10, step=1, label="Profundidad máxima (max_depth)")
        btn4 = gr.Button("Entrenar")
        met = gr.Dataframe(label="Métricas")
        cm_plot = gr.Plot()
        btn4.click(entrenar_rf, [pct, arb, prof], [met, cm_plot])


if __name__ == "__main__":
    app.launch()
