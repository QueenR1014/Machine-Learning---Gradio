"""Módulo 2: modelos de clasificación (pre-entrenados) - falla del equipo."""
import gradio as gr
import pandas as pd

from app import graficos
from app.componentes import campos_registro, tabla_metricas
from app.estilos import aviso, banner, seccion
from app.recursos import (CLASES, DESCRIPTIVE_COLS, OBJ_CLF, UMBRAL, Recursos, etiqueta, fila_aleatoria_prueba,
                          guardar_csv_temporal, leer_lote, predecir_clasificacion, registro_desde_inputs)


def construir_tab(R: Recursos):
    metricas = R.clf["metricas"]
    preds = R.clf["predicciones"]
    nombres = R.nombres_clf
    estilo, mejor = tabla_metricas(metricas, "F1", 2)
    fila_mejor = metricas.loc[mejor]

    with gr.Tab("2 · Clasificación", elem_classes=["modulo", "mod2"]):
        gr.HTML(banner(2, "Modelo de clasificación: falla del equipo",
                       "Regresión logística, Random Forest y red neuronal, entrenados previamente y evaluados "
                       "sobre el mismo conjunto de prueba (20 %)."))

        # ---- Métricas ----
        gr.HTML(seccion("Comparación de métricas",
                        "Conjunto de prueba común. La fila resaltada es el mejor modelo según F1, "
                        "porque las clases están desbalanceadas (≈75 % / 25 %)."))
        gr.Dataframe(estilo, interactive=False)
        gr.HTML(aviso(f"Mejor modelo: <b>{fila_mejor['Modelo']}</b>, con F1 = {fila_mejor['F1']:.3f}, "
                      f"accuracy = {fila_mejor['Accuracy']:.3f} y recall = {fila_mejor['Recall']:.3f}."))

        gr.HTML(seccion("Matrices de confusión"))
        gr.Plot(graficos.matrices_confusion(
            preds["y_real"], {k: preds[k] for k in nombres if k in preds}, nombres, modulo=2), show_label=False)

        # ---- Predicción individual ----
        gr.HTML(seccion("Predecir un nuevo registro",
                        "Ingresa las seis variables del equipo. Los tres modelos predicen con el mismo registro "
                        f"y la clase Falla se asigna cuando la probabilidad es ≥ {UMBRAL:.2f}."))
        campos = campos_registro(R)
        with gr.Row():
            btn_ejemplo = gr.Button("Cargar un registro del conjunto de prueba", variant="secondary")
            btn_predecir = gr.Button("Predecir", variant="primary")
        nota_ejemplo = gr.HTML()
        with gr.Row():
            with gr.Column(scale=1):
                veredicto = gr.HTML()
                tabla_pred = gr.Dataframe(interactive=False, label="Predicciones")
            with gr.Column(scale=1):
                grafico_pred = gr.Plot(show_label=False)

        def cargar_ejemplo():
            fila = fila_aleatoria_prueba(preds, R.datos)
            real = CLASES[int(fila[OBJ_CLF])]
            return [*(float(fila[c]) for c in DESCRIPTIVE_COLS),
                    aviso(f"Registro <b>#{fila.name}</b> del conjunto de prueba. Valor real: <b>{real}</b>.")]

        def predecir(*valores):
            try:
                X = registro_desde_inputs(valores)
            except ValueError as e:
                return aviso(str(e), error=True), None, None
            probs = predecir_clasificacion(R, X).iloc[0]
            filas = [{"Modelo": nombres.get(k, k),
                      "Predicción": CLASES[int(p >= UMBRAL)],
                      "Probabilidad de falla": f"{p:.1%}"} for k, p in probs.items()]
            n_falla = int((probs >= UMBRAL).sum())
            texto = (f"<b>{n_falla} de {len(probs)}</b> modelos predicen <b>falla</b> para este equipo."
                     if n_falla else "Ningún modelo predice falla para este equipo.")
            return (f'<div class="aviso veredicto">{texto}</div>', pd.DataFrame(filas),
                    graficos.barras_probabilidad({nombres.get(k, k): p for k, p in probs.items()}, UMBRAL))

        btn_ejemplo.click(cargar_ejemplo, None, [*campos, nota_ejemplo]).then(
            predecir, campos, [veredicto, tabla_pred, grafico_pred])
        btn_predecir.click(predecir, campos, [veredicto, tabla_pred, grafico_pred])
        # Si el usuario edita los campos a mano, ya no es el registro de prueba
        for c in campos:
            c.input(lambda: "", None, nota_ejemplo)

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
            probs = predecir_clasificacion(R, X)
            salida = X.rename(columns=etiqueta).copy()
            for k in probs.columns:
                salida[f"{nombres.get(k, k)} - prob."] = probs[k].round(4)
                salida[f"{nombres.get(k, k)} - clase"] = (probs[k] >= UMBRAL).map(lambda b: CLASES[int(b)])
            msg = f"{len(X):,} registros predichos."
            if len(descartados):
                msg += f" {len(descartados):,} filas descartadas por valores faltantes o no numéricos."
            return aviso(msg), salida, guardar_csv_temporal(salida, "predicciones_clasificacion_lote.csv")

        btn_lote.click(predecir_lote, archivo, [estado_lote, tabla_lote, descarga])
