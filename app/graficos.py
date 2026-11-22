"""Gráficas de la app (matplotlib sin pyplot: cada Figure se libera sola, sin estado global)."""
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from app.estilos import PALETAS
from app.recursos import CLASES, etiqueta

TINTA = "#1f2937"
GRIS = "#9aa5b1"


def _fig(ancho=7.0, alto=4.0, ncols=1, nrows=1):
    fig = Figure(figsize=(ancho, alto), dpi=110)
    axes = fig.subplots(nrows, ncols)
    return fig, axes


def _estilo(ax, grid_y=True):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#cbd2d9")
    ax.tick_params(colors=TINTA, labelsize=9)
    if grid_y:
        ax.grid(axis="y", color="#e5e7eb", linewidth=.8)
        ax.set_axisbelow(True)


def cmap_secuencial(n: int) -> LinearSegmentedColormap:
    oscuro, medio, claro = PALETAS[n]
    return LinearSegmentedColormap.from_list(f"mod{n}", ["#ffffff", claro, medio, oscuro])


# ---------------- Módulo 1 ----------------

def histograma(datos: pd.DataFrame, col: str, bins: int, por_falla: bool, obj_clf: str):
    oscuro, medio, claro = PALETAS[1]
    fig, ax = _fig()
    if por_falla:
        for valor, color in ((0, medio), (1, oscuro)):
            ax.hist(datos.loc[datos[obj_clf] == valor, col], bins=bins, alpha=.65, color=color,
                    edgecolor="white", linewidth=.6, label=CLASES[valor])
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.hist(datos[col], bins=bins, color=medio, edgecolor="white", linewidth=.6)
        ax.axvline(datos[col].mean(), color=oscuro, linestyle="--", linewidth=1.4,
                   label=f"Media = {datos[col].mean():,.2f}")
        ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Histograma de {etiqueta(col)}", fontsize=12, fontweight="bold", color=TINTA, loc="left")
    ax.set_xlabel(etiqueta(col), color=TINTA)
    ax.set_ylabel("Frecuencia", color=TINTA)
    _estilo(ax)
    fig.tight_layout()
    return fig


def boxplot(datos: pd.DataFrame, col: str, por_falla: bool, obj_clf: str):
    oscuro, medio, claro = PALETAS[1]
    fig, ax = _fig()
    if por_falla:
        grupos = [datos.loc[datos[obj_clf] == v, col] for v in (0, 1)]
        etiquetas, colores = [CLASES[0], CLASES[1]], [claro, medio]
    else:
        grupos, etiquetas, colores = [datos[col]], [etiqueta(col)], [claro]
    bp = ax.boxplot(grupos, vert=False, patch_artist=True, widths=.55,
                    medianprops=dict(color=oscuro, linewidth=2),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=medio, markeredgecolor="none", alpha=.6),
                    whiskerprops=dict(color=oscuro), capprops=dict(color=oscuro))
    for caja, color in zip(bp["boxes"], colores):
        caja.set_facecolor(color)
        caja.set_edgecolor(oscuro)
    ax.set_yticks(range(1, len(etiquetas) + 1), etiquetas)
    ax.set_title(f"Boxplot de {etiqueta(col)}", fontsize=12, fontweight="bold", color=TINTA, loc="left")
    ax.set_xlabel(etiqueta(col), color=TINTA)
    _estilo(ax, grid_y=False)
    ax.grid(axis="x", color="#e5e7eb", linewidth=.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def correlaciones(datos: pd.DataFrame):
    oscuro, medio, claro = PALETAS[1]
    corr = datos.corr(numeric_only=True)
    cmap = LinearSegmentedColormap.from_list("div", [GRIS, "#eef1f4", "#ffffff", claro, oscuro])
    fig, ax = _fig(7.2, 5.6)
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    nombres = [etiqueta(c) for c in corr.columns]
    ax.set_xticks(range(len(nombres)), nombres, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(nombres)), nombres, fontsize=8.5)
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if v > .6 else TINTA)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    ax.set_title("Mapa de correlaciones (Pearson)", fontsize=12, fontweight="bold", color=TINTA, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def balance_clases(datos: pd.DataFrame, obj_clf: str):
    oscuro, medio, claro = PALETAS[1]
    conteo = datos[obj_clf].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = _fig(5.2, 3.6)
    barras = ax.bar([CLASES[0], CLASES[1]], conteo.values, color=[claro, medio], edgecolor=oscuro, width=.55)
    total = conteo.sum()
    for b, v in zip(barras, conteo.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / total:.1%})", ha="center", va="bottom",
                fontsize=9, color=TINTA)
    ax.set_ylim(0, conteo.max() * 1.25)
    ax.set_title("Falla del equipo", fontsize=12, fontweight="bold", color=TINTA, loc="left")
    ax.set_ylabel("Registros", color=TINTA)
    _estilo(ax)
    fig.tight_layout()
    return fig


# ---------------- Módulos 2 y 4 ----------------

def matrices_confusion(y_true, predicciones: dict, nombres: dict, modulo: int):
    """predicciones: {titulo_o_llave: array 0/1}."""
    n = len(predicciones)
    fig, axes = _fig(4.1 * n, 3.9, ncols=n)
    axes = np.atleast_1d(axes)
    cmap = cmap_secuencial(modulo)
    for ax, (key, pred) in zip(axes, predicciones.items()):
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max() * 1.05)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14, fontweight="bold",
                        color="#ffffff" if cm[i, j] > cm.max() * .55 else TINTA)
        ax.set_xticks([0, 1], [CLASES[0], CLASES[1]], fontsize=9)
        ax.set_yticks([0, 1], [CLASES[0], CLASES[1]], fontsize=9)
        ax.set_xlabel("Predicción", color=TINTA)
        ax.set_ylabel("Real", color=TINTA)
        ax.set_title(nombres.get(key, key), fontsize=11.5, fontweight="bold", color=TINTA)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    return fig


def barras_probabilidad(probs: dict, umbral: float):
    """probs: {nombre_modelo: probabilidad de falla}."""
    oscuro, medio, claro = PALETAS[2]
    fig, ax = _fig(6.4, 3.0)
    nombres = list(probs)
    valores = [probs[n] for n in nombres]
    colores = [medio if v >= umbral else claro for v in valores]
    barras = ax.barh(nombres, valores, color=colores, edgecolor=oscuro, height=.55)
    ax.axvline(umbral, color=oscuro, linestyle="--", linewidth=1.3)
    ax.text(umbral, len(nombres) - .45, f" umbral {umbral:.2f}", color=oscuro, fontsize=8.5, va="bottom")
    for b, v in zip(barras, valores):
        ax.text(min(v + .02, .9), b.get_y() + b.get_height() / 2, f"{v:.1%}", va="center", fontsize=9.5,
                color=TINTA, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("Probabilidad de falla por modelo", fontsize=12, fontweight="bold", color=TINTA, loc="left")
    ax.set_xlabel("Probabilidad de falla", color=TINTA)
    _estilo(ax, grid_y=False)
    fig.tight_layout()
    return fig


def importancias(modelo, columnas, modulo: int):
    oscuro, medio, claro = PALETAS[modulo]
    imp = pd.Series(modelo.feature_importances_, index=[etiqueta(c) for c in columnas]).sort_values()
    fig, ax = _fig(5.6, 3.6)
    ax.barh(imp.index, imp.values, color=[claro] * (len(imp) - 1) + [medio], edgecolor=oscuro, height=.6)
    for i, v in enumerate(imp.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8.5, color=TINTA)
    ax.set_xlim(0, imp.max() * 1.2)
    ax.set_title("Importancia de las variables", fontsize=12, fontweight="bold", color=TINTA, loc="left")
    _estilo(ax, grid_y=False)
    fig.tight_layout()
    return fig


def coeficientes(coefs, columnas, modulo: int, titulo: str):
    """Barras de coeficientes (sobre variables escaladas: comparables entre sí)."""
    oscuro, medio, claro = PALETAS[modulo]
    serie = pd.Series(np.ravel(coefs), index=[etiqueta(c) for c in columnas])
    serie = serie.reindex(serie.abs().sort_values().index)
    fig, ax = _fig(5.6, 3.6)
    ax.barh(serie.index, serie.values, color=[medio if v >= 0 else claro for v in serie.values],
            edgecolor=oscuro, height=.6)
    ax.axvline(0, color=oscuro, linewidth=1)
    for i, v in enumerate(serie.values):
        ax.text(v, i, f" {v:.3f} " if v >= 0 else f" {v:.3f} ", va="center", fontsize=8.5, color=TINTA,
                ha="left" if v >= 0 else "right")
    lim = serie.abs().max() * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_title(titulo, fontsize=12, fontweight="bold", color=TINTA, loc="left")
    _estilo(ax, grid_y=False)
    fig.tight_layout()
    return fig


def curva_entrenamiento(historia: dict, modulo: int, metrica_extra: str = None):
    """Pérdida de entrenamiento y validación por época (historia = history.history de Keras)."""
    oscuro, medio, claro = PALETAS[modulo]
    fig, ax = _fig(5.6, 3.6)
    epocas = np.arange(1, len(historia["loss"]) + 1)
    ax.plot(epocas, historia["loss"], color=medio, linewidth=1.8, label="Entrenamiento")
    if "val_loss" in historia:
        ax.plot(epocas, historia["val_loss"], color=oscuro, linewidth=1.8, linestyle="--", label="Validación")
    ax.set_yscale("log")
    ax.set_xlabel("Época", color=TINTA)
    ax.set_ylabel("Pérdida (escala log)", color=TINTA)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Curva de entrenamiento", fontsize=12, fontweight="bold", color=TINTA, loc="left")
    _estilo(ax)
    fig.tight_layout()
    return fig


def residuos(y_real, y_pred, modulo: int):
    oscuro, medio, claro = PALETAS[modulo]
    res = np.asarray(y_pred) - np.asarray(y_real)
    fig, ax = _fig(5.6, 3.6)
    ax.hist(res, bins=40, color=medio, edgecolor="white", linewidth=.6)
    ax.axvline(0, color=oscuro, linestyle="--", linewidth=1.3)
    ax.set_xlabel("Predicho − real (kWh)", color=TINTA)
    ax.set_ylabel("Frecuencia", color=TINTA)
    ax.set_title(f"Distribución de residuos (media {res.mean():+.3f})", fontsize=12, fontweight="bold",
                 color=TINTA, loc="left")
    _estilo(ax)
    fig.tight_layout()
    return fig


def dispersion_real_predicho(y_real, y_pred, titulo: str, r2: float, modulo: int):
    oscuro, medio, claro = PALETAS[modulo]
    y, p = np.asarray(y_real), np.asarray(y_pred)
    lim = [min(y.min(), p.min()) - 5, max(y.max(), p.max()) + 5]
    fig, ax = _fig(5.2, 3.9)
    ax.scatter(y, p, s=9, alpha=.5, color=medio, edgecolors="none")
    ax.plot(lim, lim, color=oscuro, linestyle="--", linewidth=1.3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title(f"{titulo}\nR² = {r2:.4f}", fontsize=11, fontweight="bold", color=TINTA)
    ax.set_xlabel("Consumo real (kWh)", color=TINTA)
    ax.set_ylabel("Consumo predicho (kWh)", color=TINTA)
    _estilo(ax, grid_y=False)
    ax.grid(color="#eef1f4")
    fig.tight_layout()
    return fig


# ---------------- Módulo 3 ----------------

def real_vs_predicho(predicciones: pd.DataFrame, nombres: dict, metricas: dict):
    """predicciones: columnas y_real + una por modelo."""
    oscuro, medio, claro = PALETAS[3]
    modelos = [c for c in predicciones.columns if c != "y_real"]
    y = predicciones["y_real"].to_numpy()
    lim = [y.min() - 5, y.max() + 5]
    fig, axes = _fig(4.2 * len(modelos), 4.0, ncols=len(modelos))
    for ax, key in zip(np.atleast_1d(axes), modelos):
        ax.scatter(y, predicciones[key], s=9, alpha=.5, color=medio, edgecolors="none")
        ax.plot(lim, lim, color=oscuro, linestyle="--", linewidth=1.3)
        nombre = nombres.get(key, key)
        ax.set_title(f"{nombre}\nR² = {metricas[nombre]:.4f}", fontsize=11, fontweight="bold", color=TINTA)
        ax.set_xlabel("Consumo real (kWh)", color=TINTA)
        ax.set_ylabel("Consumo predicho (kWh)", color=TINTA)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        _estilo(ax, grid_y=False)
        ax.grid(color="#eef1f4")
    fig.tight_layout()
    return fig


def barras_consumo(valores: dict):
    oscuro, medio, claro = PALETAS[3]
    fig, ax = _fig(6.4, 3.0)
    nombres = list(valores)
    vals = [valores[n] for n in nombres]
    barras = ax.barh(nombres, vals, color=[claro, medio, oscuro][:len(nombres)], height=.55)
    for b, v in zip(barras, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:,.2f} kWh", va="center", fontsize=9.5,
                color=TINTA, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.3)
    ax.invert_yaxis()
    ax.set_title("Consumo predicho por modelo", fontsize=12, fontweight="bold", color=TINTA, loc="left")
    ax.set_xlabel("Consumo energético predicho (kWh)", color=TINTA)
    _estilo(ax, grid_y=False)
    fig.tight_layout()
    return fig
