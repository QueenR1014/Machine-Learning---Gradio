"""
Respaldo de resultados/ (modelos, métricas y gráficas).

- respaldo/optimizados/: copia protegida de los modelos entrenados con optimize=True (~2 h).
  Solo se escribe a propósito (python entrenar.py --respaldar); entrenar nunca la toca.
- respaldo/ultimo/: copia automática de resultados/ justo antes de cada entrenamiento,
  para poder deshacer un entrenamiento accidental.
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.training import RESULTADOS, TAREAS, _archivos

DIR_RESPALDO = ROOT / "respaldo"
RESPALDOS = {
    "optimizados": DIR_RESPALDO / "optimizados",
    "ultimo": DIR_RESPALDO / "ultimo",
}
NOTA = "LEEME.txt"


def _rel(archivo: Path) -> Path:
    return archivo.relative_to(RESULTADOS)


def _necesarios() -> list:
    """Archivos que guarda train_all_models (los mismos que revisa modelos_listos)."""
    return [f for tarea in TAREAS for f in _archivos(tarea)]


def _extras() -> list:
    """Archivos de resultados/ que no son obligatorios para la app pero conviene conservar (historiales)."""
    return sorted(f for f in RESULTADOS.rglob("*") if f.is_file() and f not in _necesarios())


def respaldo_completo(nombre: str) -> bool:
    destino = RESPALDOS[nombre]
    return all((destino / _rel(f)).is_file() for f in _necesarios())


def copiar(origen: Path, destino: Path, archivos: list):
    for f in archivos:
        objetivo = destino / f.relative_to(origen)
        objetivo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, objetivo)


def crear_respaldo(nombre: str = "optimizados", descripcion: str = "") -> Path:
    """Copia resultados/ -> respaldo/<nombre>/ (reemplaza lo que hubiera)."""
    faltan = [f.name for f in _necesarios() if not f.is_file()]
    if faltan:
        raise FileNotFoundError(f"resultados/ está incompleto (faltan {faltan}); no hay nada que respaldar.")
    destino = RESPALDOS[nombre]
    if destino.exists():
        shutil.rmtree(destino)
    copiar(RESULTADOS, destino, _necesarios() + _extras())
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    (destino / NOTA).write_text(f"Copia de resultados/ hecha el {fecha}.\n{descripcion}\n", encoding="utf-8")
    return destino


def restaurar_respaldo(nombre: str = "optimizados") -> Path:
    """Copia respaldo/<nombre>/ -> resultados/ (sobrescribe los modelos actuales)."""
    origen = RESPALDOS[nombre]
    if not respaldo_completo(nombre):
        raise FileNotFoundError(f"No hay un respaldo completo en {origen.relative_to(ROOT)}.")
    archivos = [f for f in origen.rglob("*") if f.is_file() and f.name != NOTA]
    copiar(origen, RESULTADOS, archivos)
    return origen


def describir(nombre: str) -> str:
    nota = RESPALDOS[nombre] / NOTA
    if not respaldo_completo(nombre):
        return "no existe"
    return nota.read_text(encoding="utf-8").strip().replace("\n", " ") if nota.is_file() else "completo"
