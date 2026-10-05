"""Pruebas del módulo de reporte y de la interfaz interactiva."""

import io
import sympy as sp
import pytest

from lagrange.analisis import analizar, Configuracion
from lagrange.entrada import parsear_funcion, parsear_lista
from lagrange.interfaz import Interfaz, SalirPrograma, Cancelado
from lagrange.reporte import reporte_texto, reporte_markdown

R = sp.Rational


def test_reporte_completo_ejemplo_b():
    f = parsear_funcion("1/x")
    nodos = parsear_lista("2, 2.75, 4")
    puntos = [sp.Integer(3)]
    an = analizar(nodos, funcion=f, puntos=puntos)

    texto = reporte_texto(an)
    assert "POLINOMIO DE INTERPOLACIÓN DE LAGRANGE" in texto
    assert "1. Datos" in texto
    assert "2. Polinomios base de Lagrange" in texto
    assert "3. Polinomio de Lagrange" in texto
    assert "4. Verificaciones" in texto
    assert "5. Interpolación en x* = 3" in texto
    assert "6. Interpolación sin el polinomio" in texto
    assert "7. Errores en x* = 3" in texto
    assert "8. Cota teórica del error" in texto
    assert "9. Resumen final (x* = 3)" in texto

    # Comprobar presencia de valores clave exactos de Test B
    assert "P_2(x) = x^2/22 - 35*x/88 + 49/44" in texto
    assert "29/88" in texto
    assert "1/264" in texto
    assert "1/64" in texto
    assert "9/256" in texto
    assert "se cumple (OK)" in texto


def test_reporte_markdown_generacion():
    f = parsear_funcion("x^2")
    nodos = parsear_lista("0, 1, 2")
    puntos = [sp.Integer(1)]
    an = analizar(nodos, funcion=f, puntos=puntos)
    md = reporte_markdown(an)
    assert "# Polinomio de interpolación de Lagrange" in md
    assert "```text" in md


def test_reporte_modo_datos():
    nodos = parsear_lista("0, 2, 5")
    valores = parsear_lista("18, 24, 21")
    puntos = [sp.Integer(3)]
    reales = [sp.Integer(23)]
    an = analizar(nodos, valores=valores, puntos=puntos, reales=reales)
    texto = reporte_texto(an)
    assert "Modo: tabla de datos" in texto
    assert "LA COTA NO SE PUEDE CALCULAR" in texto
    assert "Error absoluto" in texto


def test_interfaz_cargar_ejemplo():
    salidas = []
    ui = Interfaz(leer=lambda _: "0", escribir=lambda s: salidas.append(s))
    ui.opcion_ejemplo()
    assert ui.analisis is not None
    assert ui.analisis.lagrange.n == 2
    assert ui.puntos == [sp.Integer(3)]


def test_interfaz_cancelar_reintento():
    entradas = iter(["no_un_numero", "cancelar"])
    ui = Interfaz(leer=lambda _: next(entradas), escribir=lambda _: None)
    with pytest.raises(Cancelado):
        ui.pedir("Ingrese: ", lambda t: int(t))


def test_interfaz_despachar_salir():
    ui = Interfaz(leer=lambda _: "0", escribir=lambda _: None)
    with pytest.raises(SalirPrograma):
        ui.despachar("0")
