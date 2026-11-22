"""
Módulo 4: entrenamiento interactivo de cualquiera de los seis modelos (clasificación o regresión).

El usuario elige el modelo, el % de entrenamiento y sus hiperparámetros (los controles se generan a partir de
config/model_options.json). Al entrenar:
  1. config.configManager.update_config() valida y escribe esos valores en models/model_cfg.json
  2. training.construir_estructura() hace el split con ese test_size y escala los datos
  3. la función del backend entrena con optimize=False, leyendo model_cfg.json
Todo bajo CFG_LOCK para que dos usuarios no se pisen la configuración.
"""
import time

import gradio as gr
import pandas as pd

from app import graficos
from app.estilos import aviso, banner, kpis, seccion
from app.recursos import DESCRIPTIVE_COLS, Recursos
from config.configManager import CFG_LOCK, editable_params, test_size_spec, update_config

# llave del backend -> (tarea, llave en model_cfg.json, hiperparámetros principales)
MODELOS = {
    "LogReg":     ("classification", "LogReg",        ["C", "max_iter"]),
    "RandForest": ("classification", "RandomForest",  ["n_estimators", "max_depth"]),
    "NNC":        ("classification", "NeuralNetwork", ["hidden_layer_sizes", "max_iter"]),
    "LinearReg":  ("regression",     "LinearReg",     []),
    "SVR":        ("regression",     "SVR",           ["C", "epsilon"]),
    "NNR":        ("regression",     "NeuralNetwork", ["hidden_layer_sizes", "max_iter"]),
}
MODELO_INICIAL = "RandForest"
TAREA_ES = {"classification": "Clasificación", "regression": "Regresión"}
METRICAS = {"classification": ["Accuracy", "Recall", "Specificity", "Precision", "F1"],
            "regression": ["R2", "MSE", "RMSE", "MAE"]}
MENOR_ES_MEJOR = {"MSE", "RMSE", "MAE"}
DECIMALES = {"classification": 3, "regression": 4}   # como en training.calcular_metricas_*
SEG_POR_EPOCA = 0.3   # medido en este equipo con los datos del proyecto
COLUMNAS_HIST = ["#", "Modelo", "% entrenamiento", "Hiperparámetros", "Resultado", "Tiempo (s)"]


# ---------------- Controles generados desde model_options.json ----------------

def _texto_opcion(c) -> str:
    return "Ninguno (todas)" if c is None else str(c)


def _control(param: str, spec: dict, cfg_modelo: str):
    """
    Crea el control de un hiperparámetro según su tipo en model_options.json.
    Retorna (componentes, valores_iniciales, convertir) donde convertir(*valores) da el valor para update_config.
    """
    label = spec.get("label", param)
    inicial = spec.get("ui_default", spec["default"])
    tipo = spec["type"]

    if tipo == "int" and spec.get("nullable"):
        sin_limite = gr.Checkbox(inicial is None, label=f"{label}: sin límite")
        valor = inicial if inicial is not None else min(10, spec["max"])
        slider = gr.Slider(spec["min"], spec["max"], value=valor, step=spec.get("step", 1), label=label,
                           interactive=inicial is not None)
        sin_limite.change(lambda s: gr.update(interactive=not s), sin_limite, slider)
        return ([sin_limite, slider], [inicial is None, gr.update(value=valor, interactive=inicial is not None)],
                lambda s, v: None if s else v)

    if tipo == "int":
        info = None
        if cfg_modelo == "NeuralNetwork" and param == "max_iter":
            info = (f"≈ {SEG_POR_EPOCA} s por época. El valor de model_cfg.json es {spec['default']} "
                    f"(≈ {spec['default'] * SEG_POR_EPOCA / 60:.0f} min).")
        slider = gr.Slider(spec["min"], spec["max"], value=inicial, step=spec.get("step", 1), label=label, info=info)
        return [slider], [inicial], lambda v: v

    if tipo == "float":
        numero = gr.Number(value=inicial, label=label, minimum=spec.get("min"), maximum=spec.get("max"),
                           info=f"Entre {spec.get('min')} y {spec.get('max')}")
        return [numero], [inicial], lambda v: v

    if tipo == "choice":
        menu = gr.Dropdown([(_texto_opcion(c), str(c)) for c in spec["choices"]], value=str(inicial), label=label)
        return [menu], [str(inicial)], lambda v: None if v == "None" else v

    if tipo == "bool":
        return [gr.Checkbox(bool(inicial), label=label)], [bool(inicial)], lambda v: v

    if tipo == "int_list":
        texto = ", ".join(map(str, inicial))
        caja = gr.Textbox(texto, label=label,
                          info=f"Neuronas por capa, separadas por coma ({spec.get('min_len', 1)} a "
                               f"{spec.get('max_len')} capas, {spec.get('min')}–{spec.get('max')} neuronas).")
        return [caja], [texto], lambda v: v

    raise ValueError(f"Tipo '{tipo}' sin control en la app ({param}).")


def _texto_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items()) or "—"


# ---------------- Tab ----------------

def construir_tab(R: Recursos):
    from models.classificationModels import (LogClassification, NNC, NOMBRES as NOMBRES_CLF, RandForest,
                                             classification_metrics, probabilidad_falla)
    from models.regressionModels import (LinearReg, NNR, NOMBRES as NOMBRES_REG, SVR, predecir_consumo,
                                         regression_metrics)
    from training.training import DIR_METRICAS, construir_estructura

    nombres = {**NOMBRES_CLF, **NOMBRES_REG}
    entrenadores = {
        "LogReg": lambda ds: LogClassification(ds, optimize=False),
        "RandForest": lambda ds: RandForest(ds, optimize=False),
        "NNC": lambda ds: NNC(ds, optimize=False),
        "LinearReg": lambda ds: LinearReg(ds),
        "SVR": lambda ds: SVR(ds, optimize=False),
        "NNR": lambda ds: NNR(ds, optimize=False),
    }

    def tarea_de(key):
        return MODELOS[key][0]

    def resultados(key):
        return R.clf if tarea_de(key) == "classification" else R.reg

    def metricas_base(key) -> pd.Series:
        tabla = resultados(key)["metricas"].set_index("Modelo")
        return tabla.loc[nombres[key], METRICAS[tarea_de(key)]]

    def descripcion_base(key) -> str:
        """Hiperparámetros del modelo pre-entrenado del módulo 2/3 (leídos del modelo guardado)."""
        modelo = resultados(key)["modelos"][key]
        if hasattr(modelo, "get_params"):
            if key == "LinearReg":
                return "Mínimos cuadrados ordinarios (sin hiperparámetros)"
            p = modelo.get_params()
            return _texto_params({k: p[k] for k in MODELOS[key][2] if k in p})
        capas = [l.units for l in modelo.layers if hasattr(l, "units")][:-1]
        ruta = DIR_METRICAS / f"historial_{key}.csv"
        epocas = len(pd.read_csv(ruta)) if ruta.is_file() else "?"
        return f"capas ocultas={capas}, épocas={epocas}"

    def tabla_inicial(key) -> pd.DataFrame:
        base = metricas_base(key)
        return pd.DataFrame({"Métrica": base.index, "Pre-entrenado": base.values,
                             "Nuevo modelo": "—", "Diferencia": "—", "¿Mejora?": "—"})

    sp_test = test_size_spec()
    pct_defecto = round((1 - sp_test["default"]) * 100)
    opciones_modelo = [(f"{TAREA_ES[t]} · {nombres[k]}", k) for k, (t, _, _) in MODELOS.items()]

    with gr.Tab("4 · Interactivo", elem_classes=["modulo", "mod4"]):
        gr.HTML(banner(4, "Entrena tu propio modelo",
                       "Elige cualquiera de los seis modelos, ajusta el porcentaje de entrenamiento y sus "
                       "hiperparámetros, entrena y compara contra la versión pre-entrenada de los módulos 2 y 3."))

        with gr.Row():
            # ================= Controles =================
            with gr.Column(scale=1, min_width=320):
                gr.HTML(seccion("Configuración"))
                modelo_sel = gr.Dropdown(opciones_modelo, value=MODELO_INICIAL, label="Modelo")
                pct_train = gr.Slider(round((1 - sp_test["max"]) * 100), round((1 - sp_test["min"]) * 100),
                                      value=pct_defecto, step=round(sp_test["step"] * 100),
                                      label="Porcentaje de entrenamiento (%)",
                                      info="El resto se usa como conjunto de prueba.")

                grupos, componentes, iniciales, controles = {}, [], [], {}
                for key, (tarea, cfg_key, principales) in MODELOS.items():
                    specs = editable_params(tarea, cfg_key)
                    otros = [p for p in specs if p not in principales]
                    controles[key] = []
                    with gr.Group(visible=key == MODELO_INICIAL) as grupos[key]:
                        gr.HTML(aviso(f"<b>Pre-entrenado:</b> {descripcion_base(key)}"))
                        if not specs:
                            gr.HTML(aviso("La regresión lineal se resuelve por mínimos cuadrados: no tiene "
                                          "hiperparámetros. Solo cambia el porcentaje de entrenamiento."))

                        def agregar(params):
                            for p in params:
                                comps, vals, convertir = _control(p, specs[p], cfg_key)
                                controles[key].append((p, slice(len(componentes), len(componentes) + len(comps)),
                                                       convertir))
                                componentes.extend(comps)
                                iniciales.extend(vals)

                        agregar(principales)
                        if otros:
                            with gr.Accordion("Otros hiperparámetros", open=False):
                                agregar(otros)

                with gr.Row():
                    btn_defecto = gr.Button("Valores por defecto", variant="secondary")
                    btn_entrenar = gr.Button("Entrenar modelo", variant="primary")
                gr.HTML(aviso("Si cambias el porcentaje de entrenamiento, el conjunto de prueba también cambia y "
                              "la comparación con el pre-entrenado (80 % / 20 %) es aproximada."))

            # ================= Resultados =================
            with gr.Column(scale=2):
                gr.HTML(seccion("Resultados de la evaluación"))
                estado = gr.HTML(aviso("Elige un modelo, ajusta los parámetros y presiona <b>Entrenar modelo</b>."))
                tabla_comp = gr.Dataframe(tabla_inicial(MODELO_INICIAL), interactive=False,
                                          label="Métricas en el conjunto de prueba")
                with gr.Row():
                    grafico_1 = gr.Plot(show_label=False)
                    grafico_2 = gr.Plot(show_label=False)

        gr.HTML(seccion("Historial de entrenamientos", "Entrenamientos hechos en esta sesión."))
        tabla_hist = gr.Dataframe(pd.DataFrame(columns=COLUMNAS_HIST), interactive=False, wrap=True)
        historial = gr.State([])

        # ================= Lógica =================
        def cambiar_modelo(key):
            return ([gr.update(visible=k == key) for k in MODELOS]
                    + [aviso(f"Modelo seleccionado: <b>{nombres[key]}</b> ({TAREA_ES[tarea_de(key)].lower()}). "
                             "Presiona <b>Entrenar modelo</b>."), tabla_inicial(key), None, None])

        modelo_sel.change(cambiar_modelo, modelo_sel,
                          [*grupos.values(), estado, tabla_comp, grafico_1, grafico_2])

        btn_defecto.click(lambda: [pct_defecto, *iniciales], None, [pct_train, *componentes])

        def entrenar(key, pct, hist, progress=gr.Progress(track_tqdm=True), *valores):
            tarea, cfg_key, _ = MODELOS[key]
            test_size = round(1 - pct / 100, 4)
            elegidos = {p: convertir(*valores[sl]) for p, sl, convertir in controles[key]}
            vacio = (hist, pd.DataFrame(hist, columns=COLUMNAS_HIST).iloc[::-1])
            try:
                with CFG_LOCK:
                    cfg = update_config(tarea, cfg_key, elegidos, test_size=test_size)
                    t0 = time.perf_counter()
                    ds = construir_estructura(tarea, test_size=test_size)
                    salida = entrenadores[key](ds)
                    segundos = time.perf_counter() - t0
            except ValueError as e:
                return aviso(str(e), error=True), gr.update(), None, None, *vacio

            modelo, history = salida if isinstance(salida, tuple) else (salida, None)
            aplicados = {p: cfg[tarea][cfg_key][p] for p in editable_params(tarea, cfg_key)}

            # ---- métricas sobre el conjunto de prueba ----
            if tarea == "classification":
                pred = (probabilidad_falla(salida, ds.X_test_scaled) >= 0.5).astype(int)
                nuevas = pd.Series(classification_metrics(ds.y_test, pred))[METRICAS[tarea]]
            else:
                pred = predecir_consumo(salida, ds.X_test_scaled)
                nuevas = pd.Series(regression_metrics(ds.y_test, pred))[METRICAS[tarea]]
            # Mismo redondeo con el que training.py guardó las métricas del pre-entrenado
            dec = DECIMALES[tarea]
            nuevas = nuevas.astype(float).round(dec)
            base = metricas_base(key)
            dif = (nuevas.values - base.values).round(dec)
            mejora = [("✓" if (d < 0 if m in MENOR_ES_MEJOR else d > 0) else "✗" if d != 0 else "=")
                      for m, d in zip(nuevas.index, dif)]
            comp = pd.DataFrame({"Métrica": nuevas.index, "Pre-entrenado": base.values,
                                 "Nuevo modelo": nuevas.values, "Diferencia": dif, "¿Mejora?": mejora})

            # ---- gráficas ----
            if tarea == "classification":
                g1 = graficos.matrices_confusion(ds.y_test, {nombres[key]: pred}, {}, modulo=4)
                principal = [(f"{nuevas['F1']:.3f}", "F1"), (f"{nuevas['Accuracy']:.3f}", "Accuracy")]
                resultado = f"F1 {nuevas['F1']:.3f} · Acc {nuevas['Accuracy']:.3f}"
            else:
                g1 = graficos.dispersion_real_predicho(ds.y_test, pred, nombres[key], nuevas["R2"], modulo=4)
                principal = [(f"{nuevas['R2']:.4f}", "R²"), (f"{nuevas['RMSE']:.3f}", "RMSE (kWh)")]
                resultado = f"R² {nuevas['R2']:.4f} · RMSE {nuevas['RMSE']:.3f}"

            if history is not None:
                g2 = graficos.curva_entrenamiento(history.history, modulo=4)
            elif hasattr(modelo, "feature_importances_"):
                g2 = graficos.importancias(modelo, DESCRIPTIVE_COLS, modulo=4)
            elif hasattr(modelo, "coef_"):
                g2 = graficos.coeficientes(modelo.coef_, DESCRIPTIVE_COLS, 4,
                                           "Coeficientes (variables estandarizadas)")
            else:
                g2 = graficos.residuos(ds.y_test, pred, modulo=4)

            estado_html = kpis(principal + [
                (f"{len(ds.X_train):,} / {len(ds.X_test):,}", "Registros entrenamiento / prueba"),
                (f"{segundos:.1f} s", "Tiempo de entrenamiento"),
            ])
            hist = hist + [{
                "#": len(hist) + 1, "Modelo": f"{TAREA_ES[tarea]} · {nombres[key]}", "% entrenamiento": int(pct),
                "Hiperparámetros": _texto_params(aplicados), "Resultado": resultado,
                "Tiempo (s)": round(segundos, 1),
            }]
            return estado_html, comp, g1, g2, hist, pd.DataFrame(hist, columns=COLUMNAS_HIST).iloc[::-1]

        btn_entrenar.click(
            lambda: (aviso("Entrenando…"), gr.update(interactive=False)), None, [estado, btn_entrenar]
        ).then(
            entrenar, [modelo_sel, pct_train, historial, *componentes],
            [estado, tabla_comp, grafico_1, grafico_2, historial, tabla_hist],
        ).then(lambda: gr.update(interactive=True), None, btn_entrenar)
