"""Pruebas de la interpolación sin el polinomio (R4)."""

import sympy as sp

from lagrange.entrada import evaluar_funcion_en_nodos, parsear_funcion, parsear_lista
from lagrange.interpolacion import evaluacion_directa, iguales, neville
from lagrange.polinomio import construir_lagrange, evaluar_polinomio

R = sp.Rational


def _datos(f: str, nodos: str):
    xs = parsear_lista(nodos)
    return xs, evaluar_funcion_en_nodos(parsear_funcion(f), xs)


def test_B_directa():
    xs, ys = _datos("1/x", "2, 2.75, 4")
    res = evaluacion_directa(xs, ys, sp.Integer(3))
    assert res.valor == R(29, 88)
    assert res.suma_l == 1
    assert [t.lk for t in res.terminos] == [R(-1, 6), R(16, 15), R(1, 10)]


def test_B_neville():
    xs, ys = _datos("1/x", "2, 2.75, 4")
    res = neville(xs, ys, sp.Integer(3))
    assert res.valor == R(29, 88)
    # Q_{1,1} y Q_{2,1} son las rectas por (x0,x1) y (x1,x2) evaluadas en 3.
    assert res.tabla[1][1] == R(7, 22)
    assert res.tabla[2][1] == R(15, 44)
    assert len(res.pasos) == 3


def test_neville_es_el_mismo_polinomio_en_cada_entrada():
    """Q_{i,j} es el polinomio que interpola x_{i-j},...,x_i evaluado en x*."""
    xs, ys = _datos("exp(x)", "0, 0.5, 1, 1.5")
    xstar = R(7, 10)
    res = neville(xs, ys, xstar)
    for i in range(len(xs)):
        assert res.tabla[i][0] == ys[i]
        for j in range(1, i + 1):
            sub = construir_lagrange(xs[i - j : i + 1], ys[i - j : i + 1])
            assert iguales(res.tabla[i][j], evaluar_polinomio(sub.polinomio, xstar))


def test_tres_metodos_coinciden():
    for f, nodos, xstar in [
        ("sin(x)", "0, 0.4, 1.1, 2", R(3, 4)),
        ("sqrt(x)", "1, 2, 4, 7, 9", R(5)),
        ("x^3 - 2*x", "-2, -1, 1, 3", R(1, 2)),
    ]:
        xs, ys = _datos(f, nodos)
        p = construir_lagrange(xs, ys).polinomio
        vp = evaluar_polinomio(p, xstar)
        assert iguales(evaluacion_directa(xs, ys, xstar).valor, vp)
        assert iguales(neville(xs, ys, xstar).valor, vp)


def test_xstar_en_un_nodo():
    xs, ys = _datos("1/x", "2, 2.75, 4")
    assert evaluacion_directa(xs, ys, R(11, 4)).valor == R(4, 11)
    assert neville(xs, ys, R(11, 4)).valor == R(4, 11)


def test_nodos_desordenados():
    xs, ys = _datos("1/x", "4, 2, 2.75")
    assert evaluacion_directa(xs, ys, sp.Integer(3)).valor == R(29, 88)
    assert neville(xs, ys, sp.Integer(3)).valor == R(29, 88)


def test_modo_datos():
    xs, ys = parsear_lista("0, 2, 5"), parsear_lista("18, 24, 21")
    p = construir_lagrange(xs, ys).polinomio
    assert evaluacion_directa(xs, ys, 3).valor == neville(xs, ys, 3).valor == evaluar_polinomio(p, 3)
