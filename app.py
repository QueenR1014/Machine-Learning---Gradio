"""
Aplicación Gradio: exploración de datos, clasificación (falla), regresión (consumo) y módulo interactivo.

Ejecutar desde la raíz del proyecto:
    python app.py
Requiere los modelos entrenados en resultados/ (python -m training.training).
"""
import gradio as gr

from config.configManager import reset_config
from app.estilos import CSS, CURSO, TEMA, creditos, encabezado
from app.recursos import Recursos
from app import modulo1_exploracion, modulo2_clasificacion, modulo3_regresion, modulo4_interactivo


def construir_app() -> gr.Blocks:
    # Cada construcción de la app parte de la configuración por defecto
    reset_config()
    R = Recursos()

    with gr.Blocks(title=f"Mantenimiento predictivo · {CURSO}") as demo:
        gr.HTML(encabezado())
        with gr.Tabs():
            modulo1_exploracion.construir_tab(R)
            modulo2_clasificacion.construir_tab(R)
            modulo3_regresion.construir_tab(R)
            modulo4_interactivo.construir_tab(R)
        gr.HTML(creditos())
    return demo


if __name__ == "__main__":
    construir_app().launch(theme=TEMA, css=CSS)
