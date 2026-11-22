"""
Aplicación Gradio: exploración de datos, clasificación (falla), regresión (consumo) y módulo interactivo.

Ejecutar desde la raíz del proyecto:
    python app.py
Usa los modelos pre-entrenados de resultados/. Si no están, los restaura de respaldo/
y, si tampoco hay respaldo, entrena con optimize=False (ver entrenar.py).
"""
import gradio as gr

from config.configManager import reset_config
from app.estilos import CSS, CURSO, TEMA, creditos, encabezado
from app.recursos import Recursos
from app import modulo1_exploracion, modulo2_clasificacion, modulo3_regresion, modulo4_interactivo
from training.respaldo import respaldo_completo, restaurar_respaldo
from training.training import modelos_listos, train_all_models


def asegurar_modelos():
    """Deja resultados/ completo: respaldo optimizado > último respaldo > entrenamiento rápido."""
    if modelos_listos():
        return
    for nombre in ("optimizados", "ultimo"):
        if respaldo_completo(nombre):
            restaurar_respaldo(nombre)
            print(f"No había modelos en resultados/: se restauraron desde respaldo/{nombre}/")
            return
    print("No hay modelos ni respaldos: entrenando con optimize=False (~20 minutos)...")
    train_all_models(optimize=False)


def construir_app() -> gr.Blocks:
    # Cada construcción de la app parte de la configuración por defecto
    reset_config()
    asegurar_modelos()
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
