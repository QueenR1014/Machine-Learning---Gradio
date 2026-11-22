"""Módulo 3: modelos de regresión (pre-entrenados) - consumo energético."""
import gradio as gr
import pandas as pd

from app import graficos, hiperparametros
from app.componentes import campos_registro, tabla_metricas
from app.estilos import aviso, banner, seccion
from app.recursos import (DESCRIPTIVE_COLS, OBJ_REG, Recursos, etiqueta, fila_aleatoria_prueba,
                          guardar_csv_temporal, leer_lote, predecir_regresion, registro_desde_inputs)


def construir_tab(R: Recursos):
    metricas = R.reg["metricas"]
    preds = R.reg["predicciones"]
    nombres = R.nombres_reg
    estilo, mejor = tabla_metricas(metricas, "RMSE", 3, mayor_es_mejor=False)
    fila_mejor = metricas.loc[mejor]
    r2_por_modelo = dict(zip(metricas["Modelo"], metricas["R2"]))

    with gr.Tab("3 – Regresión", elem_classes=["modulo", "mod3"]):
        gr.HTML(banner(3, "Modelo de regresión: consumo energético",
                       "Regresión lineal, SVR y red neuronal, entrenados previamente y evaluados sobre el "
                       "mismo conjunto de prueba (20 %). El consumo se predice en kWh."))

        # ---- Métricas ----
        gr.HTML(seccion("Comparación de métricas",
                        "R² (más alto es mejor); MSE, RMSE y MAE (más bajo es mejor). "
                        "La fila resaltada es el modelo con menor RMSE."))
        gr.Dataframe(estilo, interactive=False)
        gr.HTML(aviso(f"Mejor modelo: <b>{fila_mejor['Modelo']}</b>, con R² = {fila_mejor['R2']:.4f}, "
                      f"RMSE = {fila_mejor['RMSE']:.3f} kWh y MAE = {fila_mejor['MAE']:.3f} kWh."))

        gr.HTML(seccion("Hiperparámetros de los modelos cargados",
                        "Configuración con la que se entrenó cada modelo que usa este módulo."))
        gr.HTML(hiperparametros.tarjetas(R.reg["modelos"], nombres))

        gr.HTML(seccion("Consumo real vs predicho", "Cada punto es un equipo del conjunto de prueba; "
                                                    "la línea punteada es la predicción perfecta."))
        gr.Plot(graficos.real_vs_predicho(preds[["y_real", *[k for k in nombres if k in preds]]],
                                          nombres, r2_por_modelo), show_label=False)

        # ---- Predicción individual ----
        gr.HTML(seccion("Predecir el consumo de un nuevo registro",
                        "Ingresa las seis variables del equipo para ver la predicción de cada modelo."))
        campos = campos_registro(R)
        with gr.Row():
            btn_ejemplo = gr.Button("Cargar un registro del conjunto de prueba", variant="secondary")
            btn_predecir = gr.Button("Predecir consumo", variant="primary")
        nota_ejemplo = gr.HTML()
        with gr.Row():
            with gr.Column(scale=1):
                resumen = gr.HTML()
                tabla_pred = gr.Dataframe(interactive=False, label="Predicciones")
            with gr.Column(scale=1):
                grafico_pred = gr.Plot(show_label=False)

        real_actual = gr.State(None)

        def cargar_ejemplo():
            fila = fila_aleatoria_prueba(preds, R.datos)
            real = float(fila[OBJ_REG])
            return [*(float(fila[c]) for c in DESCRIPTIVE_COLS),
                    aviso(f"Registro <b>#{fila.name}</b> del conjunto de prueba. "
                          f"Consumo real: <b>{real:,.2f} kWh</b>."), real]

        def predecir(*args):
            *valores, real = args
            try:
                X = registro_desde_inputs(valores)
            except ValueError as e:
                return aviso(str(e), error=True), None, None
            consumo = predecir_regresion(R, X).iloc[0]
            filas = []
            for k, v in consumo.items():
                fila = {"Modelo": nombres.get(k, k), "Consumo predicho (kWh)": round(float(v), 2)}
                if real is not None:
                    fila["Error vs real (kWh)"] = round(float(v) - real, 2)
                filas.append(fila)
            texto = (f"Consumo promedio de los tres modelos: <b>{consumo.mean():,.2f} kWh</b> "
                     f"(rango {consumo.min():,.2f} – {consumo.max():,.2f}).")
            return (f'<div class="aviso veredicto">{texto}</div>', pd.DataFrame(filas),
                    graficos.barras_consumo({nombres.get(k, k): float(v) for k, v in consumo.items()}))

        salidas = [resumen, tabla_pred, grafico_pred]
        btn_ejemplo.click(cargar_ejemplo, None, [*campos, nota_ejemplo, real_actual]).then(
            predecir, [*campos, real_actual], salidas)
        # Si el usuario edita los campos a mano, ya no es el registro de prueba: se olvida el valor real
        for c in campos:
            c.input(lambda: (None, ""), None, [real_actual, nota_ejemplo])
        btn_predecir.click(predecir, [*campos, real_actual], salidas)

        # ---- Lote ----
        with gr.Accordion("Predecir varios registros desde un archivo (CSV o Excel)", open=False):
            gr.Markdown("El archivo debe tener las columnas "
                        + ", ".join(f"**{etiqueta(c)}**" for c in DESCRIPTIVE_COLS)
                        + ". También se aceptan los nombres sin tildes. Las columnas adicionales se ignoran.")
            with gr.Row():
                archivo = gr.File(file_types=[".csv", ".xlsx", ".xls"], label="Archivo")
                descarga = gr.File(label="Resultados (descargar)", interactive=False)
            btn_lote = gr.Button("Predecir archivo", variant="primary")
            estado_lote = gr.HTML()
            tabla_lote = gr.Dataframe(interactive=False, max_height=420)

        def predecir_lote(ruta):
            if not ruta:
                return aviso("Primero sube un archivo.", error=True), None, None
            try:
                original, X, descartados = leer_lote(ruta)
            except (ValueError, FileNotFoundError) as e:
                return aviso(str(e), error=True), None, None
            consumo = predecir_regresion(R, X)
            salida = X.rename(columns=etiqueta).copy()
            for k in consumo.columns:
                salida[f"{nombres.get(k, k)} (kWh)"] = consumo[k].round(2)
            msg = f"{len(X):,} registros predichos."
            if len(descartados):
                msg += f" {len(descartados):,} filas descartadas por valores faltantes o no numéricos."
            return aviso(msg), salida, guardar_csv_temporal(salida, "predicciones_regresion_lote.csv")

        btn_lote.click(predecir_lote, archivo, [estado_lote, tabla_lote, descarga])
