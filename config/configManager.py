"""
Gestión de model_cfg.json a partir de model_options.json.

- model_options.json: catálogo de parámetros. Para cada uno guarda su valor por defecto ("default")
  y, si el usuario lo puede modificar desde la app ("editable": true), su tipo y rango/opciones.
- model_cfg.json: configuración activa que leen los modelos con optimize=False (via load_params()).

Flujo de la app:
    reset_config()                      # al construir la app: model_cfg.json vuelve a los defaults
    with CFG_LOCK:                      # módulo interactivo: escribir y entrenar sin que otro usuario
        update_config("classification", "RandomForest", {"n_estimators": 300}, test_size=0.3)
        ...entrenar con optimize=False...
"""
import json
import os
import re
import tempfile
import threading
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
ROOT = CONFIG_DIR.parent
RUTA_OPCIONES = CONFIG_DIR / "model_options.json"
# model_cfg.json se queda en models/: load_params() (models/dataManagement.py) lo lee de esa carpeta
RUTA_CFG = ROOT / "models" / "model_cfg.json"

TAREAS = ("classification", "regression")

# Reentrante: la app puede tomarlo alrededor de update_config() + entrenamiento
CFG_LOCK = threading.RLock()


# ---------------- Lectura ----------------

def load_options() -> dict:
    with open(RUTA_OPCIONES, "r", encoding="utf-8") as f:
        opciones = json.load(f)
    opciones.pop("_info", None)
    return opciones


def default_config(opciones: dict = None) -> dict:
    """model_cfg con todos los valores por defecto de model_options.json."""
    opciones = opciones or load_options()
    cfg = {"test_size": opciones["test_size"]["default"]}
    for tarea in TAREAS:
        cfg[tarea] = {
            modelo: {param: spec["default"] for param, spec in params.items()}
            for modelo, params in opciones[tarea].items()
        }
    return cfg


def load_config() -> dict:
    with open(RUTA_CFG, "r", encoding="utf-8") as f:
        return json.load(f)


def editable_params(tarea: str, modelo: str) -> dict:
    """Solo los parámetros que la app puede mostrar como controles: {param: spec}."""
    params = _specs_modelo(load_options(), tarea, modelo)
    return {p: spec for p, spec in params.items() if spec.get("editable")}


def test_size_spec() -> dict:
    return load_options()["test_size"]


# ---------------- Escritura ----------------

def reset_config() -> dict:
    """Sobrescribe model_cfg.json con los defaults. Llamar al construir la app."""
    cfg = default_config()
    with CFG_LOCK:
        _escribir(cfg)
    return cfg


def update_config(tarea: str, modelo: str, valores: dict = None, test_size: float = None) -> dict:
    """
    Valida y escribe en model_cfg.json los valores del usuario para un modelo (y opcionalmente test_size).
    Los parámetros no incluidos conservan su valor actual. Lanza ValueError si algo no es válido;
    en ese caso model_cfg.json no se modifica.
    """
    opciones = load_options()
    specs = _specs_modelo(opciones, tarea, modelo)

    limpios = {}
    for param, valor in (valores or {}).items():
        if param not in specs:
            raise ValueError(f"'{param}' no es un parámetro de {tarea}/{modelo}. Opciones: {list(specs)}")
        spec = specs[param]
        if not spec.get("editable"):
            raise ValueError(f"'{param}' de {tarea}/{modelo} no es editable.")
        limpios[param] = validar_valor(spec, valor, nombre=param)

    with CFG_LOCK:
        cfg = _config_actual_o_default(opciones)
        if test_size is not None:
            cfg["test_size"] = validar_valor(opciones["test_size"], test_size, nombre="test_size")
        cfg[tarea][modelo].update(limpios)
        _escribir(cfg)
    return cfg


# ---------------- Validación ----------------

def validar_valor(spec: dict, valor, nombre: str = "valor"):
    """Convierte y valida un valor según su spec. Retorna el valor listo para guardar en el JSON."""
    tipo = spec["type"]

    if valor is None or (isinstance(valor, str) and valor.strip() == "" and tipo != "choice"):
        if spec.get("nullable"):
            return None
        raise ValueError(f"'{nombre}' no puede estar vacío.")

    if tipo == "fixed":
        return spec["default"]

    if tipo == "bool":
        if isinstance(valor, str):
            v = valor.strip().lower()
            if v in ("true", "1", "si", "sí", "yes"):
                return True
            if v in ("false", "0", "no"):
                return False
            raise ValueError(f"'{nombre}' debe ser verdadero o falso.")
        return bool(valor)

    if tipo == "choice":
        for opcion in spec["choices"]:
            if valor == opcion or (opcion is not None and str(valor) == str(opcion)):
                return opcion
        raise ValueError(f"'{nombre}' debe ser una de {spec['choices']}, se recibió {valor!r}.")

    if tipo in ("int", "float"):
        try:
            numero = float(str(valor).replace(",", ".")) if isinstance(valor, str) else float(valor)
        except (TypeError, ValueError):
            raise ValueError(f"'{nombre}' debe ser numérico, se recibió {valor!r}.")
        if tipo == "int":
            if not numero.is_integer():
                raise ValueError(f"'{nombre}' debe ser un entero, se recibió {valor!r}.")
            numero = int(numero)
        _chequear_rango(spec, numero, nombre)
        return numero

    if tipo == "int_list":
        if isinstance(valor, str):                     # "32, 16" -> [32, 16]
            partes = [p for p in re.split(r"[,\s;]+", valor.strip()) if p]
        else:
            partes = list(valor)
        try:
            lista = [int(p) for p in partes]
        except (TypeError, ValueError):
            raise ValueError(f"'{nombre}' debe ser una lista de enteros (p. ej. '32, 16').")
        if not spec.get("min_len", 1) <= len(lista) <= spec.get("max_len", len(lista)):
            raise ValueError(f"'{nombre}' debe tener entre {spec.get('min_len', 1)} y {spec.get('max_len')} capas.")
        for n in lista:
            _chequear_rango(spec, n, nombre)
        return lista

    raise ValueError(f"Tipo '{tipo}' desconocido en model_options.json ('{nombre}').")


def _chequear_rango(spec: dict, numero, nombre: str):
    if "min" in spec and numero < spec["min"]:
        raise ValueError(f"'{nombre}' debe ser >= {spec['min']}, se recibió {numero}.")
    if "max" in spec and numero > spec["max"]:
        raise ValueError(f"'{nombre}' debe ser <= {spec['max']}, se recibió {numero}.")


# ---------------- Internos ----------------

def _specs_modelo(opciones: dict, tarea: str, modelo: str) -> dict:
    if tarea not in TAREAS:
        raise ValueError(f"Tarea '{tarea}' no válida. Usa {TAREAS}.")
    if modelo not in opciones[tarea]:
        raise ValueError(f"Modelo '{modelo}' no existe en {tarea}. Opciones: {list(opciones[tarea])}")
    return opciones[tarea][modelo]


def _config_actual_o_default(opciones: dict) -> dict:
    """Config actual completada con defaults (por si model_cfg.json falta, está dañado o le faltan llaves)."""
    cfg = default_config(opciones)
    try:
        actual = load_config()
    except (FileNotFoundError, json.JSONDecodeError):
        return cfg
    if "test_size" in actual:
        cfg["test_size"] = actual["test_size"]
    for tarea in TAREAS:
        for modelo, params in cfg[tarea].items():
            params.update({k: v for k, v in actual.get(tarea, {}).get(modelo, {}).items() if k in params})
    return cfg


def a_json(datos) -> str:
    """JSON con indentación 2, pero listas cortas en una sola línea ([32, 16])."""
    texto = json.dumps(datos, indent=2, ensure_ascii=False)
    return re.sub(r"\[\s*([^\[\]{}]*?)\s*\]",
                  lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]",
                  texto) + "\n"


def _escribir(cfg: dict, ruta: Path = RUTA_CFG):
    # Escritura atómica: si algo falla, model_cfg.json queda como estaba
    fd, tmp = tempfile.mkstemp(dir=ruta.parent, prefix=".model_cfg_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(a_json(cfg))
        os.replace(tmp, ruta)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    reset_config()
    print(f"{RUTA_CFG.name} reiniciado con los valores de {RUTA_OPCIONES.name}")
