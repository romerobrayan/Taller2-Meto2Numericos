"""Punto de entrada principal para el programa de interpolación de Lagrange.

Permite uso interactivo (menú en español) o ejecución directa por línea de comandos (CLI)
con argumentos para pruebas y reproducción automatizada.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lagrange.analisis import Configuracion, analizar
from lagrange.entrada import (
    ErrorEntrada,
    parsear_funcion,
    parsear_lista,
    validar_datos,
    validar_nodos,
)
from lagrange.graficas import exportar_reporte, graficar
from lagrange.interfaz import Interfaz
from lagrange.reporte import reporte_texto


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interpolación polinómica de Lagrange y análisis de error (Burden & Faires, Cap. 3).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--f",
        type=str,
        default=None,
        help="Función simbólica a interpolar (ej. '1/x', 'sin(x)', 'exp(x)*cos(x)').",
    )
    parser.add_argument(
        "--nodos",
        type=str,
        default=None,
        help="Arreglo de nodos x0, x1, ..., xn (ej. '2, 2.75, 4' o '[0, 2, 5]').",
    )
    parser.add_argument(
        "--valores",
        type=str,
        default=None,
        help="Valores y0, y1, ..., yn para modo de tabla de datos (ej. '18, 24, 21').",
    )
    parser.add_argument(
        "--x",
        type=str,
        default=None,
        help="Punto(s) x* donde interpolar (ej. '3' o '2.5, 3.5').",
    )
    parser.add_argument(
        "--y-real",
        type=str,
        default=None,
        help="Valor(es) real(es) conocido(s) en x* para calcular errores en modo datos (ej. '23').",
    )
    parser.add_argument(
        "--decimales",
        "-d",
        type=int,
        default=8,
        help="Número de cifras significativas a mostrar en decimales (por defecto: 8).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-12,
        help="Tolerancia para el método de bisección (por defecto: 1e-12).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Iteraciones máximas para el método de bisección (por defecto: 100).",
    )
    parser.add_argument(
        "--var",
        type=str,
        default=None,
        help="Variable independiente (ej. 'x', 't', 'u'). Por defecto: detectada de --f o 'x'.",
    )
    parser.add_argument(
        "--solo-resumen",
        action="store_true",
        help="Muestra únicamente la sección 9 de resumen final consolidado.",
    )
    parser.add_argument(
        "--graficar",
        "-g",
        nargs="?",
        const="",
        default=None,
        help="Genera gráfica de f(x), P_n(x) y error. Opcionalmente indique la ruta del archivo PNG.",
    )
    parser.add_argument(
        "--exportar",
        "-e",
        nargs="?",
        const="",
        default=None,
        help="Exporta el reporte a un archivo Markdown. Opcionalmente indique la ruta de salida.",
    )
    parser.add_argument(
        "--interactivo",
        "-i",
        action="store_true",
        help="Inicia el menú interactivo en español (por defecto si no se pasan argumentos).",
    )
    return parser


def ejecutar_cli(args: argparse.Namespace) -> int:
    try:
        if args.nodos is None:
            print(
                "Error: Se deben especificar los nodos mediante --nodos "
                "(o use sin argumentos para el menú interactivo).",
                file=sys.stderr,
            )
            return 1

        nodos = validar_nodos(parsear_lista(args.nodos))
        f_expr = parsear_funcion(args.f) if args.f is not None else None

        valores = None
        if args.valores is not None:
            valores = parsear_lista(args.valores)
            validar_datos(nodos, valores)

        if f_expr is None and valores is None:
            print(
                "Error: Debe ingresar una función f(x) con --f o una lista de valores con --valores.",
                file=sys.stderr,
            )
            return 1

        puntos = parsear_lista(args.x) if args.x is not None else []
        reales = None
        if args.y_real is not None:
            reales = parsear_lista(args.y_real)

        cfg = Configuracion(
            cifras=args.decimales,
            tol=args.tol,
            max_iter=args.max_iter,
        )

        var_sym = sp.Symbol(args.var.strip()) if args.var is not None else None

        an = analizar(
            nodos=nodos,
            valores=valores,
            funcion=f_expr,
            puntos=puntos,
            reales=reales,
            config=cfg,
            var=var_sym,
        )

        # Imprimir reporte
        print(reporte_texto(an, solo_resumen=args.solo_resumen))

        # Graficar si se solicitó
        if args.graficar is not None:
            ruta_grafica = args.graficar.strip() if args.graficar.strip() else None
            archivo_g = graficar(an, ruta=ruta_grafica)
            print(f"\n[Gráfica guardada en: {archivo_g}]")

        # Exportar si se solicitó
        if args.exportar is not None:
            ruta_exp = args.exportar.strip() if args.exportar.strip() else None
            archivo_e = exportar_reporte(an, ruta=ruta_exp, solo_resumen=args.solo_resumen)
            print(f"\n[Reporte exportado en: {archivo_e}]")

        return 0

    except ErrorEntrada as exc:
        print(f"\nError de entrada: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"\nError durante la ejecución: {exc}", file=sys.stderr)
        return 3


def main() -> None:
    parser = crear_parser()
    # Si no se pasan argumentos por línea de comandos o se pasa --interactivo
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ("-i", "--interactivo")):
        ui = Interfaz()
        ui.ejecutar()
    else:
        args = parser.parse_args()
        if args.interactivo:
            ui = Interfaz()
            ui.ejecutar()
        else:
            codigo = ejecutar_cli(args)
            if codigo != 0:
                sys.exit(codigo)


if __name__ == "__main__":
    main()
