"""Paletas de cada módulo, tema y CSS de la app."""
import gradio as gr

# (oscuro, medio, claro) por módulo
PALETAS = {
    1: ("#0957C3", "#3A8DEF", "#8ACAFF"),   # Exploración
    2: ("#002042", "#006293", "#54BAE2"),   # Clasificación
    3: ("#00382F", "#219B89", "#85D3C6"),   # Regresión
    4: ("#31215E", "#6A67BA", "#9997D3"),   # Interactivo
}

CURSO = "Aprendizaje Automático de Máquinas"
INTEGRANTES = [
    "Luz Marina Cortés Rico",
    "Andrés Jesús Ospino Stevenson",
    "Juan José Reina Reyes",
]

# Fuente del sistema (SF Pro en Apple, Segoe UI en Windows): ya trae tamaño óptico y tracking ajustado
FUENTE_SISTEMA = ["-apple-system", "BlinkMacSystemFont", "SF Pro Text", "system-ui", "Segoe UI",
                  "Roboto", "Helvetica Neue", "sans-serif"]

TEMA = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=FUENTE_SISTEMA,
    font_mono=["SF Mono", "ui-monospace", "Cascadia Mono", "Consolas", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
)


def _css_modulo(n: int) -> str:
    oscuro, medio, claro = PALETAS[n]
    return f"""
.mod{n} {{
    --m-oscuro: {oscuro};
    --m-medio: {medio};
    --m-claro: {claro};
}}
button[role="tab"]:nth-of-type({n})[aria-selected="true"] {{
    color: {oscuro} !important;
}}
.dark button[role="tab"]:nth-of-type({n})[aria-selected="true"] {{
    color: {claro} !important;
}}
button[role="tab"]:nth-of-type({n})[aria-selected="true"]::after {{
    background: {medio} !important;
}}
"""


CSS = "".join(_css_modulo(n) for n in PALETAS) + """
/* overflow: clip (no hidden) para que la barra de pestañas pueda quedarse fija con position: sticky */
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; overflow: clip !important; }

/* Variables de Gradio redefinidas dentro de cada módulo -> botones, sliders, checkboxes con su paleta */
.modulo {
    --button-primary-background-fill: var(--m-medio);
    --button-primary-background-fill-hover: var(--m-oscuro);
    --button-primary-border-color: var(--m-medio);
    --button-primary-border-color-hover: var(--m-oscuro);
    --button-primary-text-color: #ffffff;
    --button-primary-text-color-hover: #ffffff;
    --button-secondary-border-color: var(--m-claro);
    --slider-color: var(--m-medio);
    --color-accent: var(--m-medio);
    --color-accent-soft: color-mix(in srgb, var(--m-claro) 25%, transparent);
    --checkbox-background-color-selected: var(--m-medio);
    --checkbox-border-color-selected: var(--m-medio);
    --checkbox-border-color-focus: var(--m-medio);
    --input-border-color-focus: var(--m-medio);
    --loader-color: var(--m-medio);
    --block-label-background-fill: var(--m-medio);
    --block-label-text-color: #ffffff;
    --block-label-border-color: var(--m-medio);
    --block-title-background-fill: var(--m-medio);
    --block-title-text-color: #ffffff;
    --block-title-border-color: var(--m-medio);
    --block-info-text-color: var(--body-text-color-subdued);
}

.banner {
    background: linear-gradient(135deg, var(--m-oscuro) 0%, var(--m-medio) 100%);
    color: #ffffff;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 6px;
}
.banner h2 { margin: 0 0 6px 0; font-size: 1.45rem; color: #ffffff; }
.banner p { margin: 0; opacity: .92; font-size: .95rem; color: #ffffff; }
.banner .num {
    display: inline-block; background: rgba(255,255,255,.18); border-radius: 999px;
    padding: 2px 12px; font-size: .78rem; letter-spacing: .04em; margin-bottom: 8px;
}

.seccion {
    border-left: 4px solid var(--m-medio);
    padding: 2px 0 2px 12px;
    margin: 18px 0 4px 0;
}
.seccion h3 { margin: 0; font-size: 1.08rem; }
.seccion p { margin: 2px 0 0 0; opacity: .75; font-size: .88rem; }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.kpi {
    border: 1px solid color-mix(in srgb, var(--m-claro) 60%, transparent);
    background: color-mix(in srgb, var(--m-claro) 14%, transparent);
    border-radius: 12px; padding: 12px 16px;
}
.kpi .v { font-size: 1.5rem; font-weight: 700; color: var(--m-medio); line-height: 1.2; }
.kpi .l { font-size: .8rem; opacity: .75; }

.aviso {
    border-radius: 10px; padding: 10px 14px; font-size: .92rem;
    background: color-mix(in srgb, var(--m-claro) 18%, transparent);
    border: 1px solid color-mix(in srgb, var(--m-claro) 55%, transparent);
}
.aviso.error { background: #fdecea; border-color: #f5c2c0; color: #8a1c14; }
.veredicto { font-size: 1.05rem; }
.veredicto b { color: var(--m-medio); }

/* ================= Encabezado y créditos ================= */
.header-app { text-align: center; padding: 28px 16px 18px; }
.header-app .curso {
    display: inline-block; font-size: .75rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: var(--body-text-color-subdued); margin-bottom: 10px;
}
.header-app h1 {
    font-size: clamp(1.9rem, 4.2vw, 2.9rem); line-height: 1.05; letter-spacing: -.025em;
    font-weight: 700; margin: 0 0 10px 0;
}
.header-app .sub {
    font-size: 1.02rem; line-height: 1.5; color: var(--body-text-color-subdued); margin: 0 auto 18px;
    max-width: 46ch;
}
.equipo { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
.persona {
    display: inline-flex; align-items: center; gap: 8px; padding: 5px 14px 5px 5px; border-radius: 999px;
    font-size: .86rem; font-weight: 500; letter-spacing: .005em;
    background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);
}
.persona .ini {
    width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center;
    font-size: .68rem; font-weight: 700; letter-spacing: .02em; color: #fff;
}
.creditos {
    text-align: center; font-size: .82rem; line-height: 1.6; color: var(--body-text-color-subdued);
    padding: 26px 16px 8px; border-top: 1px solid var(--border-color-primary); margin-top: 28px;
}
.creditos b { color: var(--body-text-color); font-weight: 600; }

/* ================= Tipografía: tracking según el tamaño ================= */
.banner h2 { letter-spacing: -.02em; line-height: 1.15; font-weight: 700; }
.banner p { line-height: 1.5; }
.seccion h3 { letter-spacing: -.01em; line-height: 1.3; font-weight: 650; }
.seccion p { line-height: 1.5; }
.kpi .v { letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
.kpi .l { letter-spacing: .01em; }

/* ================= Profundidad: tarjetas con sombra en capas ================= */
.banner { box-shadow: 0 1px 2px rgba(0,0,0,.06), 0 8px 24px -8px color-mix(in srgb, var(--m-oscuro) 55%, transparent); }
.kpi { transition: transform 160ms ease-out; }

/* ================= Barra de pestañas translúcida y fija ================= */
.tab-wrapper {
    position: sticky; top: 0; z-index: 50;
    background: color-mix(in srgb, var(--background-fill-primary) 72%, transparent);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    backdrop-filter: blur(20px) saturate(180%);
    border-radius: 0 0 12px 12px;
    box-shadow: 0 8px 16px -12px rgba(0,0,0,.25);   /* borde de scroll suave en vez de una línea dura */
}
button[role="tab"] { font-weight: 550; letter-spacing: -.005em; }

/* ================= Respuesta inmediata al presionar ================= */
button.primary, button.secondary, button[role="tab"] {
    transition: transform 100ms ease-out, background-color 150ms ease-out, color 150ms ease-out;
}
button.primary:active, button.secondary:active { transform: scale(.97); }
button[role="tab"]:active { transform: scale(.96); }

/* Al cambiar de pestaña el contenido aparece desde donde está la barra (arriba), sin rebote */
@keyframes entrar {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: none; }
}
.modulo { animation: entrar 240ms cubic-bezier(.2, .8, .2, 1); }

/* ================= Accesibilidad ================= */
@media (prefers-reduced-motion: reduce) {
    .modulo { animation: fade 160ms ease; }
    @keyframes fade { from { opacity: 0; } to { opacity: 1; } }
    button.primary:active, button.secondary:active, button[role="tab"]:active { transform: none; }
    .kpi { transition: none; }
}
@media (prefers-reduced-transparency: reduce) {
    .tab-wrapper { background: var(--background-fill-primary); backdrop-filter: none; -webkit-backdrop-filter: none; }
}
@media (prefers-contrast: more) {
    .kpi, .aviso, .persona { border-width: 2px; border-color: currentColor; }
    .tab-wrapper { background: var(--background-fill-primary); backdrop-filter: none; }
}
"""


def _iniciales(nombre: str) -> str:
    partes = nombre.split()
    return (partes[0][0] + partes[-2][0]).upper() if len(partes) >= 3 else "".join(p[0] for p in partes[:2]).upper()


def encabezado() -> str:
    colores = [PALETAS[1][1], PALETAS[2][1], PALETAS[3][1]]
    personas = "".join(
        f'<span class="persona"><span class="ini" style="background:{c}">{_iniciales(n)}</span>{n}</span>'
        for n, c in zip(INTEGRANTES, colores))
    return (f'<header class="header-app"><span class="curso">{CURSO}</span>'
            "<h1>Mantenimiento predictivo de equipos</h1>"
            '<p class="sub">Análisis exploratorio, clasificación de fallas y regresión del consumo energético.</p>'
            f'<div class="equipo">{personas}</div></header>')


def creditos() -> str:
    return (f'<footer class="creditos"><b>{CURSO}</b><br>'
            + " · ".join(INTEGRANTES) + "</footer>")


def banner(n: int, titulo: str, descripcion: str) -> str:
    return (f'<div class="banner"><span class="num">MÓDULO {n}</span>'
            f"<h2>{titulo}</h2><p>{descripcion}</p></div>")


def seccion(titulo: str, descripcion: str = "") -> str:
    desc = f"<p>{descripcion}</p>" if descripcion else ""
    return f'<div class="seccion"><h3>{titulo}</h3>{desc}</div>'


def kpis(items: list) -> str:
    """items: [(valor, etiqueta), ...]"""
    tarjetas = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>'
                       for v, l in items)
    return f'<div class="kpis">{tarjetas}</div>'


def aviso(texto: str, error: bool = False) -> str:
    return f'<div class="aviso{" error" if error else ""}">{texto}</div>'
