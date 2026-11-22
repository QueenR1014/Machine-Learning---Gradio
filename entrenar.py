"""
Entrenamiento de todos los modelos (clasificación y regresión) y manejo de respaldos.

    python entrenar.py                  menú interactivo
    python entrenar.py --rapido         optimize=False: parámetros por defecto de config/model_options.json (~20 min)
    python entrenar.py --optimizar      optimize=True: búsqueda de hiperparámetros (~2 horas)
    python entrenar.py --restaurar      vuelve a poner en resultados/ los modelos optimizados del respaldo
    python entrenar.py --estado         muestra qué modelos y respaldos hay

Opciones extra al entrenar:
    --desde-crudos      limpia de nuevo data/datos_crudos.csv antes de entrenar
    --test-size 0.2     proporción del conjunto de prueba
    --respaldar         al terminar, guarda el resultado como respaldo/optimizados

Antes de cada entrenamiento, lo que haya en resultados/ se copia a respaldo/ultimo/
(se recupera con: python entrenar.py --restaurar ultimo).
La app se ejecuta aparte con: python app.py
"""
import argparse
import sys
import time

from config.configManager import reset_config
from training.respaldo import RESPALDOS, crear_respaldo, describir, respaldo_completo, restaurar_respaldo
from training.training import TEST_SIZE, modelos_listos, train_all_models


def estado():
    print(f"resultados/            : {'completo' if modelos_listos() else 'incompleto o vacío'}")
    for nombre, ruta in RESPALDOS.items():
        print(f"respaldo/{nombre + '/':<13}: {describir(nombre)}")


def entrenar(optimize: bool, desde_crudos: bool = False, test_size: float = TEST_SIZE,
             respaldar: bool = False):
    # Parámetros por defecto (la app puede haber dejado otros en model_cfg.json)
    reset_config()
    if modelos_listos():
        crear_respaldo("ultimo", "Copia automática de resultados/ antes de volver a entrenar.")
        print("Modelos actuales copiados a respaldo/ultimo/")

    modo = "optimize=True (búsqueda de hiperparámetros)" if optimize else "optimize=False (parámetros por defecto)"
    print(f"Entrenando con {modo}, test_size={test_size}...")
    inicio = time.time()
    metricas = train_all_models(test_size=test_size, desde_crudos=desde_crudos, optimize=optimize)
    minutos = (time.time() - inicio) / 60

    print(f"\nListo en {minutos:.1f} min. Resultados en resultados/\n")
    print(metricas["clasificacion"].to_string(index=False))
    print()
    print(metricas["regresion"].to_string(index=False))

    if respaldar:
        descripcion = (f"Modelos entrenados con optimize={optimize}, test_size={test_size} "
                       f"({minutos:.0f} min).")
        crear_respaldo("optimizados", descripcion)
        print("\nGuardado también en respaldo/optimizados/")
    return metricas


def restaurar(nombre: str):
    origen = restaurar_respaldo(nombre)
    print(f"resultados/ restaurado desde {origen.name}/: {describir(nombre)}")


def _preguntar(texto: str) -> bool:
    return input(f"{texto} [s/N]: ").strip().lower() in ("s", "si", "sí", "y", "yes")


def menu():
    estado()
    print("""
  1) Entrenamiento rápido   (optimize=False, parámetros por defecto, ~20 minutos)
  2) Entrenamiento completo (optimize=True, búsqueda de hiperparámetros, ~2 horas)
  3) Restaurar los modelos optimizados del respaldo
  4) Restaurar el último estado (antes del último entrenamiento)
  5) Salir""")
    opcion = input("Opción: ").strip()
    if opcion == "1":
        entrenar(optimize=False)
    elif opcion == "2":
        if _preguntar("Esto tarda cerca de 2 horas. ¿Continuar?"):
            metricas = entrenar(optimize=True)
            if _preguntar("¿Reemplazar respaldo/optimizados/ con estos modelos?"):
                crear_respaldo("optimizados", "Modelos entrenados con optimize=True.")
                print("Respaldo actualizado.")
    elif opcion == "3":
        restaurar("optimizados")
    elif opcion == "4":
        restaurar("ultimo")


def main():
    p = argparse.ArgumentParser(description="Entrena los modelos o maneja sus respaldos.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    modo = p.add_mutually_exclusive_group()
    modo.add_argument("--rapido", action="store_true", help="entrenar con optimize=False")
    modo.add_argument("--optimizar", action="store_true", help="entrenar con optimize=True (~2 h)")
    modo.add_argument("--restaurar", nargs="?", const="optimizados", choices=list(RESPALDOS),
                      help="copiar un respaldo a resultados/ (por defecto: optimizados)")
    modo.add_argument("--estado", action="store_true", help="mostrar qué modelos y respaldos hay")
    p.add_argument("--desde-crudos", action="store_true", help="limpiar los datos crudos antes de entrenar")
    p.add_argument("--test-size", type=float, default=TEST_SIZE, help=f"por defecto {TEST_SIZE}")
    p.add_argument("--respaldar", action="store_true", help="guardar el resultado en respaldo/optimizados/")
    a = p.parse_args()

    if not 0 < a.test_size < 1:
        p.error("--test-size debe estar entre 0 y 1")

    if a.rapido or a.optimizar:
        entrenar(optimize=a.optimizar, desde_crudos=a.desde_crudos, test_size=a.test_size,
                 respaldar=a.respaldar)
    elif a.restaurar:
        restaurar(a.restaurar)
    elif a.estado:
        estado()
    elif sys.stdin.isatty():
        menu()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
