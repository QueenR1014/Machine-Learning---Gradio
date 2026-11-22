"""Piezas de interfaz compartidas por los módulos 2 y 3."""
import gradio as gr
import pandas as pd

from app.estilos import PALETAS
from app.recursos import DESCRIPTIVE_COLS, Recursos, etiqueta


def campos_registro(R: Recursos) -> list:
    """Seis gr.Number (uno por predictora) en dos filas. Valor inicial = mediana; info = rango observado."""
    campos = []
    for fila in (DESCRIPTIVE_COLS[:3], DESCRIPTIVE_COLS[3:]):
        with gr.Row():
            for col in fila:
                mn, med, mx = R.rangos[col]
                campos.append(gr.Number(
                    value=round(med, 2), label=etiqueta(col),
                    info=f"Rango en los datos: {mn:,.2f} – {mx:,.2f}",
                    precision=0 if col == "antiguedad" else None,
                ))
    return campos


def tabla_metricas(metricas: pd.DataFrame, criterio: str, modulo: int, mayor_es_mejor: bool = True):
    """Tabla de métricas con la fila del mejor modelo (según `criterio`) resaltada con la paleta del módulo."""
    oscuro, medio, claro = PALETAS[modulo]
    mejor = metricas[criterio].idxmax() if mayor_es_mejor else metricas[criterio].idxmin()

    def estilo_fila(fila):
        if fila.name == mejor:
            return [f"background-color: {claro}55; font-weight: 700"] * len(fila)
        return [""] * len(fila)

    numericas = metricas.select_dtypes("number").columns
    return metricas.style.apply(estilo_fila, axis=1).format({c: "{:.4f}" for c in numericas}), mejor
