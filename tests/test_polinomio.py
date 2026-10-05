"""Pruebas de los polinomios base y del polinomio de Lagrange (R3)."""

import sympy as sp

from lagrange.entrada import X, evaluar_funcion_en_nodos, parsear_funcion, parsear_lista
from lagrange.polinomio import construir_lagrange, evaluar_polinomio

R = sp.Rational


def _lagrange(f: str, nodos: str):
    xs = parsear_lista(nodos)
    return construir_lagrange(xs, evaluar_funcion_en_nodos(parsear_funcion(f), xs))


def test_A_grado_1():
    res = _lagrange("1/x", "0.1, 1.2")
    assert res.n == 1
    assert sp.expand(res.polinomio - (-R(25, 3) * X + R(65, 6))) == 0


def test_B_grado_2():
    res = _lagrange("1/x", "2, 2.75, 4")
    assert res.n == 2
    assert sp.expand(res.polinomio - (X**2 / 22 - R(35, 88) * X + R(49, 44))) == 0
    assert evaluar_polinomio(res.polinomio, 3) == R(29, 88)


def test_B_bases():
    res = _lagrange("1/x", "2, 2.75, 4")
    b0, b1, b2 = res.bases
    assert b0.denominador == R(3, 2)
    assert b1.denominador == R(-15, 16)
    assert b2.denominador == R(5, 2)
    assert sp.expand(b0.expandido - (X - R(11, 4)) * (X - 4) / R(3, 2)) == 0


def test_propiedad_delta_de_kronecker():
    res = _lagrange("sin(x)", "0, 0.5, 1.3, 2, 3")
    for b in res.bases:
        for j, xj in enumerate(res.nodos):
            assert b.expandido.subs(X, xj) == (1 if j == b.k else 0)


def test_suma_de_bases_es_uno():
    res = _lagrange("exp(x)", "-1, 0, 0.5, 2")
    assert res.suma_bases == 1
    assert res.suma_bases_ok


def test_verificaciones_en_nodos():
    res = _lagrange("1/x", "2, 2.75, 4")
    assert res.verificaciones_ok
    assert [v.p_xk for v in res.verificaciones] == [R(1, 2), R(4, 11), R(1, 4)]


def test_C_reproduce_polinomio():
    res = _lagrange("x^2", "0, 1, 2")
    assert sp.expand(res.polinomio - X**2) == 0


def test_nodos_desordenados():
    ordenado = _lagrange("1/x", "2, 2.75, 4").polinomio
    desordenado = _lagrange("1/x", "4, 2, 2.75").polinomio
    assert sp.expand(ordenado - desordenado) == 0


def test_modo_datos():
    res = construir_lagrange(parsear_lista("0, 2, 5"), parsear_lista("18, 24, 21"))
    for xk, yk in zip(res.nodos, res.valores):
        assert res.polinomio.subs(X, xk) == yk


def test_contra_sympy_interpolating_poly():
    """Validación cruzada (solo en pruebas) contra la implementación de SymPy."""
    xs = parsear_lista("-1, 0.5, 2, 3.25, 4")
    ys = evaluar_funcion_en_nodos(parsear_funcion("x^4 - 3*x + 1/x"), xs)
    mio = construir_lagrange(xs, ys).polinomio
    ref = sp.interpolating_poly(len(xs), X, xs, ys)
    assert sp.simplify(mio - ref) == 0
