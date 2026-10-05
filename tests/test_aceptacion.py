"""Pruebas de aceptación integrales y casos borde (Secciones 8 y 9 de la especificación)."""

import math
import numpy as np
import pytest
import sympy as sp

from lagrange.analisis import analizar, Configuracion
from lagrange.entrada import (
    ErrorEntrada,
    X,
    advertencias_nodos,
    parsear_funcion,
    parsear_lista,
    validar_datos,
    validar_nodos,
)
from lagrange.errores import a_float, menor_o_igual
from lagrange.interpolacion import iguales
from lagrange.polinomio import construir_lagrange

R = sp.Rational


# ==============================================================================
# 1. CASOS BORDE (Sección 8)
# ==============================================================================

def test_borde_menos_de_dos_nodos():
    with pytest.raises(ErrorEntrada, match="al menos 2 nodos"):
        validar_nodos([sp.Integer(5)])


def test_borde_arreglos_distinta_longitud():
    with pytest.raises(ErrorEntrada, match="distinto tamaño"):
        validar_datos([sp.Integer(1), sp.Integer(2)], [sp.Integer(3)])


def test_borde_nodos_repetidos():
    with pytest.raises(ErrorEntrada, match="repetido"):
        validar_nodos(parsear_lista("1, 2.5, 1"))


def test_borde_nodos_desordenados():
    """Lagrange funciona igual con nodos desordenados y el intervalo es [min, max]."""
    f = parsear_funcion("1/x")
    an_ord = analizar(parsear_lista("2, 2.75, 4"), funcion=f, puntos=[sp.Integer(3)])
    an_des = analizar(parsear_lista("4, 2, 2.75"), funcion=f, puntos=[sp.Integer(3)])

    assert sp.expand(an_ord.lagrange.polinomio - an_des.lagrange.polinomio) == 0
    assert (an_des.puntos[0].cota.a, an_des.puntos[0].cota.b) == (2, 4)
    assert not an_des.puntos[0].extrapolacion


def test_borde_xstar_igual_a_nodo():
    """x* igual a un nodo -> P_n(x*) = f(x*), error = 0."""
    f = parsear_funcion("1/x")
    nodos = parsear_lista("2, 2.75, 4")
    xstar = R(11, 4)  # nodo 2.75
    an = analizar(nodos, funcion=f, puntos=[xstar])
    pt = an.puntos[0]

    assert pt.valor_polinomio == R(4, 11)
    assert pt.error.error_absoluto == 0
    assert pt.cota_puntual == 0


def test_borde_extrapolacion():
    """x* fuera del intervalo -> advertencia de extrapolación e intervalo extendido."""
    f = parsear_funcion("1/x")
    nodos = parsear_lista("2, 2.75, 4")
    xstar = sp.Integer(5)  # fuera de [2, 4]
    an = analizar(nodos, funcion=f, puntos=[xstar])
    pt = an.puntos[0]

    assert pt.extrapolacion
    assert pt.cota.extrapolacion
    assert (pt.cota.a, pt.cota.b) == (2, 5)


def test_borde_polinomio_grado_menor_o_igual_n():
    """f(x) polinomio de grado <= n -> f^(n+1) = 0, cota = 0, error real = 0."""
    f = parsear_funcion("x^3 - 2*x + 5")
    nodos = parsear_lista("0, 1, 2, 3")  # n = 3, grado de f es 3
    an = analizar(nodos, funcion=f, puntos=[R(3, 2)])
    pt = an.puntos[0]

    assert pt.cota.derivada == 0
    assert pt.cota.M == 0
    assert pt.cota.cota_global == 0
    assert pt.cota_puntual == 0
    assert pt.error.error_absoluto == 0


def test_borde_f_no_definida_en_nodo():
    f = parsear_funcion("log(x)")
    with pytest.raises(ErrorEntrada, match="no está definida"):
        analizar(parsear_lista("0, 1, 2"), funcion=f)


def test_borde_mas_de_diez_nodos_runge():
    """Más de 10 nodos -> advertencia sobre Runge y mal condicionamiento (pero calcula)."""
    nodos = [sp.Rational(i, 2) for i in range(12)]  # 12 nodos
    an = analizar(nodos, funcion=parsear_funcion("1/(1 + x^2)"), puntos=[sp.Rational(1, 4)])
    assert len(an.advertencias) > 0
    assert "Runge" in an.advertencias[0]
    assert an.lagrange.polinomio is not None


# ==============================================================================
# 2. PRUEBAS DE ACEPTACIÓN (Sección 9)
# ==============================================================================

def test_aceptacion_A_grado_1():
    """Test A — grado 1: f = 1/x, nodos [0.1, 1.2] -> P_1(x) = -25*x/3 + 65/6."""
    f = parsear_funcion("1/x")
    nodos = parsear_lista("0.1, 1.2")
    an = analizar(nodos, funcion=f)

    esperado = -R(25, 3) * X + R(65, 6)
    assert sp.expand(an.lagrange.polinomio - esperado) == 0


def test_aceptacion_B_completo():
    """Test B — grado 2 (ejemplo principal de clase):
    f = 1/x, nodos [2, 2.75, 4], x* = 3.
    Verifica simbólicamente TODOS los valores exactos requeridos.
    """
    f = parsear_funcion("1/x")
    nodos = parsear_lista("2, 2.75, 4")
    xstar = sp.Integer(3)
    an = analizar(nodos, funcion=f, puntos=[xstar])
    pt = an.puntos[0]

    # Polinomio
    assert sp.expand(an.lagrange.polinomio - (X**2 / 22 - R(35, 88) * X + R(49, 44))) == 0

    # P_2(3) = 29/88
    assert pt.valor_polinomio == R(29, 88)
    assert pt.directo.valor == R(29, 88)
    assert pt.neville.valor == R(29, 88)
    assert pt.coincide_directo
    assert pt.coincide_neville

    # Errores en x* = 3
    assert pt.error.valor_real == R(1, 3)
    assert pt.error.error_absoluto == R(1, 264)
    # error relativo = |1/3 - 29/88| / (1/3) * 100 = (1/264)/(1/3)*100 = 100/88 %
    assert pt.error.error_relativo == R(100, 88)

    # Derivadas y M
    cota = pt.cota
    assert sp.simplify(cota.derivada - (-6 / X**4)) == 0
    assert cota.M == R(3, 8)
    assert cota.x_M == 2
    assert cota.factorial == 6
    assert cota.factor == R(1, 16)

    # g(x) y g'(x)
    assert sp.expand(cota.g - (X**3 - R(35, 4) * X**2 + R(49, 2) * X - 22)) == 0
    assert sp.expand(cota.dg - (3 * X**2 - R(35, 2) * X + R(49, 2))) == 0

    # Raíces de g' halladas por bisección: 7/3 y 7/2
    raices_g = sorted(r.raiz for r in cota.biseccion_g)
    assert len(raices_g) == 2
    assert abs(raices_g[0] - 7 / 3) < 1e-11
    assert abs(raices_g[1] - 7 / 2) < 1e-11

    # Valores en candidatos de g
    valores_cands = {c.x: c.absoluto for c in cota.candidatos_g}
    assert valores_cands[R(7, 3)] == R(25, 108)
    assert valores_cands[R(7, 2)] == R(9, 16)
    assert cota.max_g == R(9, 16)

    # Cotas
    assert cota.cota_global == R(9, 256)
    assert abs(pt.g_xstar) == R(1, 4)
    assert pt.cota_puntual == R(1, 64)

    # Cadena de desigualdades: 1/264 <= 1/64 <= 9/256
    assert menor_o_igual(R(1, 264), pt.cota_puntual)
    assert menor_o_igual(pt.cota_puntual, cota.cota_global)


def test_aceptacion_C_reproduccion_exacta():
    """Test C — reproducción exacta: f = x^2, nodos [0, 1, 2] -> P_2(x) = x^2, cota = 0, error = 0."""
    f = parsear_funcion("x^2")
    nodos = parsear_lista("0, 1, 2")
    an = analizar(nodos, funcion=f, puntos=[R(1, 2)])
    pt = an.puntos[0]

    assert sp.expand(an.lagrange.polinomio - X**2) == 0
    assert pt.cota.cota_global == 0
    assert pt.error.error_absoluto == 0


# ==============================================================================
# 3. PRUEBAS DE PROPIEDADES (Property Tests)
# ==============================================================================

@pytest.mark.parametrize(
    "expr_str, intervalo",
    [
        ("sin(x)", (0.1, 2.8)),
        ("exp(x)", (-1.5, 1.5)),
        ("sqrt(x)", (1.0, 5.0)),
        ("1/x", (1.0, 4.0)),
    ],
)
def test_propiedades_aleatorias(expr_str, intervalo):
    """Para cada función y conjunto aleatorio de nodos:
    - P_n(x_k) = f(x_k)
    - direct evaluation = Neville = expanded polynomial
    - sum L_k(x*) = 1
    - actual error <= pointwise bound <= global bound (con pequeña tolerancia)
    - validación cruzada de P_n contra sympy.interpolating_poly
    """
    rng = np.random.default_rng(42)
    f = parsear_funcion(expr_str)

    for _ in range(3):
        num_nodos = rng.integers(3, 6)
        nodos_float = np.sort(rng.uniform(intervalo[0], intervalo[1], size=num_nodos))
        # Asegurar nodos distintos
        nodos = [sp.Rational(str(round(float(v), 4))) for v in nodos_float]
        validar_nodos(nodos)

        xstar_float = rng.uniform(float(nodos[0]), float(nodos[-1]))
        xstar = sp.Rational(str(round(float(xstar_float), 4)))

        an = analizar(nodos, funcion=f, puntos=[xstar], config=Configuracion(tol=1e-10))
        pt = an.puntos[0]

        # 1. P_n(x_k) = f(x_k)
        assert an.lagrange.verificaciones_ok

        # 2. direct evaluation = Neville = expanded polynomial
        assert pt.coincide_directo
        assert pt.coincide_neville
        assert iguales(pt.directo.valor, pt.neville.valor)

        # 3. sum L_k(x*) = 1
        assert pt.directo.suma_l_ok

        # 4. actual error <= pointwise bound <= global bound
        if pt.cota.aplicable and pt.cota_puntual is not None:
            err = a_float(pt.error.error_absoluto)
            cp = a_float(pt.cota_puntual)
            cg = a_float(pt.cota.cota_global)
            tol = 1e-9 * max(1.0, cg)
            assert err <= cp + tol, f"Falla error <= cota puntual: {err} > {cp}"
            assert cp <= cg + tol, f"Falla cota puntual <= cota global: {cp} > {cg}"

        # 5. Cross-check contra sympy.interpolating_poly
        poly_ref = sp.interpolating_poly(len(nodos), X, nodos, an.valores)
        assert sp.simplify(an.lagrange.polinomio - poly_ref) == 0
