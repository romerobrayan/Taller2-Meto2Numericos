"""Generación de gráficas con matplotlib y exportación de reportes.

Guarda las imágenes y reportes en la carpeta `salidas/`.
Usa el backend 'Agg' para entornos sin servidor gráfico (servidores, WSL, CI).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from .analisis import Analisis
from .entrada import X
from .errores import a_float, funcion_numerica
from .reporte import corto, dec, reporte_markdown

CARPETA_SALIDAS = Path("salidas")


def asegurar_directorio(ruta: Path | str) -> Path:
    p = Path(ruta)
    p.mkdir(parents=True, exist_ok=True)
    return p


def exportar_reporte(
    analisis: Analisis,
    ruta: Path | str | None = None,
    solo_resumen: bool = False,
) -> str:
    """Exporta el reporte en formato Markdown."""
    asegurar_directorio(CARPETA_SALIDAS)
    if ruta is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta = CARPETA_SALIDAS / f"reporte_{ts}.md"
    else:
        ruta = Path(ruta)

    contenido = reporte_markdown(analisis, solo_resumen=solo_resumen)
    ruta.write_text(contenido, encoding="utf-8")
    return str(ruta)


def graficar(
    analisis: Analisis,
    ruta: Path | str | None = None,
    puntos_malla: int = 600,
) -> str:
    """Genera la gráfica de f(x) y P_n(x), y el panel inferior con el error absoluto."""
    asegurar_directorio(CARPETA_SALIDAS)
    if ruta is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta = CARPETA_SALIDAS / f"grafica_{ts}.png"
    else:
        ruta = Path(ruta)

    nodos_val = [a_float(xk) for xk in analisis.nodos]
    puntos_xstar = [a_float(pt.xstar) for pt in analisis.puntos]
    todos_x = nodos_val + puntos_xstar
    xmin, xmax = min(todos_x), max(todos_x)
    ancho = max(xmax - xmin, 1.0)
    margen = 0.12 * ancho
    a_plot = xmin - margen
    b_plot = xmax + margen

    xs = np.linspace(a_plot, b_plot, puntos_malla)
    fn_poly = funcion_numerica(analisis.lagrange.polinomio)
    y_poly = fn_poly(xs)

    tiene_f = analisis.funcion is not None
    if tiene_f:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]}
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax2 = None

    c = analisis.config.cifras

    # Panel superior: f(x) y P_n(x)
    if tiene_f:
        fn_f = funcion_numerica(analisis.funcion)
        y_f = fn_f(xs)
        ax1.plot(xs, y_f, label=f"$f(x) = {sp.latex(analisis.funcion)}$", color="#1f77b4", lw=2)

    ax1.plot(
        xs,
        y_poly,
        label=f"$P_{{{analisis.lagrange.n}}}(x)$ (Lagrange)",
        color="#d62728",
        ls="--",
        lw=1.8,
    )

    # Nodos
    y_nodos = [a_float(yk) for yk in analisis.valores]
    ax1.scatter(
        nodos_val,
        y_nodos,
        color="#2ca02c",
        s=60,
        zorder=5,
        label=f"Nodos ($n+1 = {len(nodos_val)}$)",
        edgecolors="black",
    )
    for k, (xk, yk) in enumerate(zip(nodos_val, y_nodos)):
        ax1.annotate(
            f"$(x_{k}, y_{k})$",
            (xk, yk),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    # Puntos x*
    for pt in analisis.puntos:
        xs_val = a_float(pt.xstar)
        yp_val = a_float(pt.valor_polinomio)
        ax1.scatter([xs_val], [yp_val], color="#9467bd", marker="^", s=90, zorder=6)
        ax1.axvline(xs_val, color="#9467bd", ls=":", alpha=0.6)
        label_xs = f"$x^*={corto(pt.xstar, c)}$"
        ax1.annotate(
            f"$P_n({corto(pt.xstar, c)})\\approx {dec(pt.valor_polinomio, 4)}$",
            (xs_val, yp_val),
            textcoords="offset points",
            xytext=(10, -15),
            fontsize=9,
            color="#581845",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8f9fa", alpha=0.8),
        )

    ax1.set_title(
        f"Interpolación de Lagrange de grado $n={analisis.lagrange.n}$",
        fontsize=13,
        pad=10,
    )
    ax1.set_ylabel("$y$", fontsize=11)
    ax1.grid(True, ls=":", alpha=0.7)
    ax1.legend(loc="best", framealpha=0.9)

    # Panel inferior: Error y cota (solo si hay f(x))
    if ax2 is not None and tiene_f:
        err_abs = np.abs(y_f - y_poly)
        ax2.plot(xs, err_abs, label="$|f(x) - P_n(x)|$ (Error real)", color="#e377c2", lw=1.8)

        # Cota global
        cota_glob = None
        for pt in analisis.puntos:
            if pt.cota.aplicable and pt.cota.cota_global is not None:
                cota_glob = a_float(pt.cota.cota_global)
                break
        if cota_glob is not None:
            ax2.axhline(
                cota_glob,
                color="#ff7f0e",
                ls="-.",
                lw=1.5,
                label=f"Cota global teórica $\\approx {cota_glob:.4g}$",
            )

        # Puntos x* en el panel de error
        for pt in analisis.puntos:
            if pt.error.error_absoluto is not None:
                err_val = a_float(pt.error.error_absoluto)
                ax2.scatter(
                    [a_float(pt.xstar)],
                    [err_val],
                    color="#d62728",
                    marker="o",
                    s=50,
                    zorder=5,
                )
                ax2.annotate(
                    f"Error en $x^*$: {err_val:.4g}",
                    (a_float(pt.xstar), err_val),
                    textcoords="offset points",
                    xytext=(10, 8),
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3cd", alpha=0.8),
                )

        ax2.set_title("Comportamiento del error y cota teórica", fontsize=11, pad=6)
        ax2.set_xlabel("$x$", fontsize=11)
        ax2.set_ylabel("Error absoluto", fontsize=11)
        ax2.grid(True, ls=":", alpha=0.7)
        ax2.legend(loc="best", framealpha=0.9)
    else:
        ax1.set_xlabel("$x$", fontsize=11)

    plt.tight_layout()
    fig.savefig(str(ruta), dpi=150)
    plt.close(fig)
    return str(ruta)
