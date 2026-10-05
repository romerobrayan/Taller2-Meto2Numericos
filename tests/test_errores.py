"""Pruebas de errores en x* (R6) y de la cota teórica del error (R5)."""

import numpy as np
import pytest
import sympy as sp

from lagrange.entrada import X, evaluar_funcion_en_nodos, parsear_funcion, parsear_lista
from lagrange.errores import (
    a_float,
    calcular_cota,
    cota_puntual,
    error_en_punto,
    intervalo_trabajo,
    menor_o_igual,
    polinomio_nodal,
)
from lagrange.interpolacion import evaluacion_directa

R = sp.Rational


@pytest.fixture(scope="module")
def cota_B():
    return calcular_cota(parsear_funcion("1/x"), parsear_lista("2, 2.75, 4"), [sp.Integer(3)])


def test_B_errores_en_punto():
    e = error_en_punto(sp.Integer(3), R(29, 88), R(1, 3))
    assert e.error_absoluto == R(1, 264)
    assert e.error_relativo == R(100, 88)
    assert abs(a_float(e.error_relativo) - 1.1363636) < 1e-7


def test_relativo_usa_valor_verdadero():
    e = error_en_punto(sp.Integer(0), R(9, 10), sp.Integer(1))
    assert e.error_relativo == 10  # |1 - 0.9| / |1| * 100, no / |0.9|


def test_relativo_indefinido_si_f_es_cero():
    e = error_en_punto(sp.Integer(0), R(1, 100), sp.Integer(0))
    assert e.error_absoluto == R(1, 100)
    assert e.error_relativo is None
    assert "no está definido" in e.nota


def test_B_derivada_y_M(cota_B):
    assert cota_B.aplicable
    assert cota_B.orden == 3 and cota_B.factorial == 6
    assert sp.simplify(cota_B.derivada - (-6 / X**4)) == 0
    assert cota_B.M == R(3, 8) and cota_B.x_M == 2
    assert cota_B.factor == R(1, 16)


def test_B_g_y_raices(cota_B):
    assert sp.expand(cota_B.g - (X**3 - R(35, 4) * X**2 + R(49, 2) * X - 22)) == 0
    assert sp.expand(cota_B.dg - (3 * X**2 - R(35, 2) * X + R(49, 2))) == 0
    raices = sorted(r.raiz for r in cota_B.biseccion_g)
    assert abs(raices[0] - 7 / 3) < 1e-11 and abs(raices[1] - 7 / 2) < 1e-11
    valores = {c.x: c.absoluto for c in cota_B.candidatos_g}
    assert valores[R(7, 3)] == R(25, 108)
    assert valores[R(7, 2)] == R(9, 16)
    assert cota_B.max_g == R(9, 16) and cota_B.x_max_g == R(7, 2)


def test_B_raices_g_prima_contra_numpy(cota_B):
    coef = [float(c) for c in sp.Poly(cota_B.dg, X).all_coeffs()]
    assert np.allclose(sorted(r.raiz for r in cota_B.biseccion_g), sorted(np.roots(coef).real), atol=1e-10)


def test_B_cotas(cota_B):
    assert cota_B.cota_global == R(9, 256)
    g3, puntual = cota_puntual(cota_B, parsear_lista("2, 2.75, 4"), sp.Integer(3))
    assert abs(g3) == R(1, 4)
    assert puntual == R(1, 64)
    assert menor_o_igual(R(1, 264), puntual) and menor_o_igual(puntual, cota_B.cota_global)


def test_C_cota_cero():
    cota = calcular_cota(parsear_funcion("x^2"), parsear_lista("0, 1, 2"), [R(1, 2)])
    assert cota.aplicable and cota.M == 0 and cota.cota_global == 0


def test_intervalo_con_extrapolacion():
    a, b, ext = intervalo_trabajo(parsear_lista("2, 2.75, 4"), [sp.Integer(5)])
    assert (a, b, ext) == (2, 5, True)
    a, b, ext = intervalo_trabajo(parsear_lista("4, 2, 2.75"), [sp.Integer(3)])
    assert (a, b, ext) == (2, 4, False)


def test_extrapolacion_extiende_la_cota():
    nodos = parsear_lista("2, 2.75, 4")
    dentro = calcular_cota(parsear_funcion("1/x"), nodos, [sp.Integer(3)])
    fuera = calcular_cota(parsear_funcion("1/x"), nodos, [sp.Integer(5)])
    assert fuera.extrapolacion and fuera.b == 5
    assert a_float(fuera.max_g) >= a_float(dentro.max_g)  # |g(5)| = 3·(9/4)·1 = 27/4 > 9/16
    assert fuera.max_g == R(27, 4)


def test_cota_no_aplica_si_no_es_continua():
    cota = calcular_cota(parsear_funcion("1/x"), parsear_lista("-1, 1, 2"), [R(1, 2)])
    assert not cota.aplicable and "no aplica" in cota.motivo
    cota = calcular_cota(parsear_funcion("sqrt(x)"), parsear_lista("0, 1, 4"), [sp.Integer(2)])
    assert not cota.aplicable  # f''' = 3/(8 x^(5/2)) no es continua en 0


def test_modo_datos_sin_cota():
    cota = calcular_cota(None, parsear_lista("0, 2, 5"), [sp.Integer(3)])
    assert not cota.aplicable and "f^(3)" in cota.motivo


def test_polinomio_nodal_tiene_n_mas_1_factores():
    g = polinomio_nodal(parsear_lista("0, 1, 2, 3"))
    assert sp.degree(g, X) == 4


def test_M_por_raiz_interior():
    """f = sin(x) en [0, 3], n = 1: f'' = -sin(x), max |f''| = 1 en x = pi/2 (raíz de f''')."""
    cota = calcular_cota(parsear_funcion("sin(x)"), parsear_lista("0, 3"), [sp.Integer(1)])
    assert abs(a_float(cota.M) - 1) < 1e-12
    assert abs(a_float(cota.x_M) - np.pi / 2) < 1e-9


def test_xstar_en_nodo_error_cero():
    nodos = parsear_lista("2, 2.75, 4")
    ys = evaluar_funcion_en_nodos(parsear_funcion("1/x"), nodos)
    v = evaluacion_directa(nodos, ys, R(11, 4)).valor
    e = error_en_punto(R(11, 4), v, R(4, 11))
    assert e.error_absoluto == 0
    cota = calcular_cota(parsear_funcion("1/x"), nodos, [R(11, 4)])
    assert cota_puntual(cota, nodos, R(11, 4))[1] == 0
