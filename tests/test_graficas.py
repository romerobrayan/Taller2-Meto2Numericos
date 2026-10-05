"""Pruebas del módulo de gráficas y exportación."""

from pathlib import Path
import sympy as sp

from lagrange.analisis import analizar
from lagrange.entrada import parsear_funcion, parsear_lista
from lagrange.graficas import exportar_reporte, graficar


def test_graficar_y_exportar(tmp_path):
    f = parsear_funcion("1/x")
    nodos = parsear_lista("2, 2.75, 4")
    puntos = [sp.Integer(3)]
    an = analizar(nodos, funcion=f, puntos=puntos)

    ruta_png = tmp_path / "grafica.png"
    ruta_md = tmp_path / "reporte.md"

    res_png = graficar(an, ruta=ruta_png)
    assert Path(res_png).exists()
    assert Path(res_png).stat().st_size > 1000

    res_md = exportar_reporte(an, ruta=ruta_md)
    assert Path(res_md).exists()
    assert "# Polinomio de interpolación de Lagrange" in Path(res_md).read_text(encoding="utf-8")


def test_graficar_modo_datos(tmp_path):
    nodos = parsear_lista("0, 2, 5")
    valores = parsear_lista("18, 24, 21")
    an = analizar(nodos, valores=valores, puntos=[sp.Integer(3)])

    ruta_png = tmp_path / "grafica_datos.png"
    res_png = graficar(an, ruta=ruta_png)
    assert Path(res_png).exists()
    assert Path(res_png).stat().st_size > 1000
