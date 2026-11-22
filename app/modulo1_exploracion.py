"""Módulo 1: exploración de los datos."""
import gradio as gr
import pandas as pd

from app import graficos
from app.estilos import banner, kpis, seccion
from app.recursos import DESCRIPTIVE_COLS, OBJ_CLF, OBJ_REG, Recursos, etiqueta


def _tabla_tipos(R: Recursos) -> pd.DataFrame:
    d = R.datos
    return pd.DataFrame({
        "Variable": [etiqueta(c) for c in d.columns],
        "Rol": ["Predictora" if c in DESCRIPTIVE_COLS else
                ("Objetivo (clasificación)" if c == OBJ_CLF else "Objetivo (regresión)") for c in d.columns],
        "Tipo": [str(t) for t in d.dtypes],
        "Valores únicos": [d[c].nunique() for c in d.columns],
        "Faltantes (crudos)": [int(R.crudos.iloc[:, i].isna().sum()) for i in range(len(d.columns))],
    })


def _describe(R: Recursos) -> pd.DataFrame:
    tabla = R.datos.describe().T.round(3)
    tabla.insert(0, "Variable", [etiqueta(c) for c in tabla.index])
    return tabla.rename(columns={"count": "n", "mean": "media", "std": "desv. est.", "min": "mín",
                                 "25%": "Q1", "50%": "mediana", "75%": "Q3", "max": "máx"})


def construir_tab(R: Recursos):
    variables = [(etiqueta(c), c) for c in DESCRIPTIVE_COLS + [OBJ_REG]]
    n_crudos, n_limpios = len(R.crudos), len(R.datos)
    tasa_falla = R.datos[OBJ_CLF].mean()

    with gr.Tab("1 – Exploración", elem_classes=["modulo", "mod1"]):
        gr.HTML(banner(1, "Exploración de los datos",
                       "Vista general del conjunto de datos de los equipos: primeras filas, estadísticas "
                       "descriptivas y distribución de cada variable."))

        gr.HTML(kpis([
            (f"{n_crudos:,}", "Registros crudos"),
            (f"{n_limpios:,}", "Registros tras limpieza"),
            (f"{n_crudos - n_limpios:,}", "Registros descartados"),
            (len(DESCRIPTIVE_COLS), "Variables predictoras"),
            (f"{tasa_falla:.1%}", "Equipos con falla"),
        ]))

        # ---- Primeras filas ----
        gr.HTML(seccion("Primeras filas", "Los datos crudos tienen los nombres originales y la falla como Sí/No; "
                                          "los limpios están listos para los modelos."))
        with gr.Row():
            fuente = gr.Radio(["Datos limpios", "Datos crudos"], value="Datos limpios", label="Conjunto", scale=1)
            n_filas = gr.Slider(5, 50, value=10, step=5, label="Número de filas", scale=2)
        tabla_head = gr.Dataframe(R.datos.head(10), interactive=False, max_height=420)

        def mostrar_head(fuente, n):
            df = R.datos if fuente == "Datos limpios" else R.crudos
            return df.head(int(n))

        fuente.change(mostrar_head, [fuente, n_filas], tabla_head)
        n_filas.change(mostrar_head, [fuente, n_filas], tabla_head)

        # ---- Estructura y estadísticas ----
        gr.HTML(seccion("Estructura y estadísticas descriptivas"))
        with gr.Row():
            with gr.Column(scale=2):
                gr.Dataframe(_tabla_tipos(R), interactive=False, label="Variables")
            with gr.Column(scale=1):
                gr.Plot(graficos.balance_clases(R.datos, OBJ_CLF), show_label=False)
        gr.Dataframe(_describe(R), interactive=False, label="Estadísticas descriptivas")

        # ---- Gráfico por variable ----
        gr.HTML(seccion("Distribución de una variable", "Elige la variable y el tipo de gráfico."))
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                variable = gr.Dropdown(variables, value=DESCRIPTIVE_COLS[0], label="Variable")
                tipo = gr.Radio(["Histograma", "Boxplot"], value="Histograma", label="Tipo de gráfico")
                bins = gr.Slider(5, 80, value=30, step=1, label="Número de intervalos (bins)")
                por_falla = gr.Checkbox(False, label="Separar por falla del equipo")
            with gr.Column(scale=3):
                grafico = gr.Plot(graficos.histograma(R.datos, DESCRIPTIVE_COLS[0], 30, False, OBJ_CLF),
                                  show_label=False)

        def graficar(col, tipo, bins, por_falla):
            if tipo == "Histograma":
                return graficos.histograma(R.datos, col, int(bins), por_falla, OBJ_CLF), gr.update(visible=True)
            return graficos.boxplot(R.datos, col, por_falla, OBJ_CLF), gr.update(visible=False)

        controles = [variable, tipo, bins, por_falla]
        for c in controles:
            c.change(graficar, controles, [grafico, bins])

        # ---- Correlaciones ----
        gr.HTML(seccion("Correlaciones entre variables"))
        gr.Plot(graficos.correlaciones(R.datos), show_label=False)
