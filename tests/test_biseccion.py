"""Pruebas del método de bisección propio."""

import math

import numpy as np
import pytest

from lagrange.biseccion import biseccion, buscar_cambios_signo, buscar_raices


def test_raiz_de_dos():
    res = biseccion(lambda t: t * t - 2, 1, 2, tol=1e-12)
    assert res.convergio
    assert abs(res.raiz - math.sqrt(2)) < 1e-11
    # (b - a)/2^N < tol  =>  N ~ log2(1/1e-12) ~ 40
    assert 38 <= len(res.iteraciones) <= 42


def test_registro_de_iteraciones():
    res = biseccion(lambda t: t**3 + 4 * t**2 - 10, 1, 2, tol=1e-4)
    primera = res.iteraciones[0]
    assert (primera.a, primera.b, primera.p) == (1.0, 2.0, 1.5)
    assert primera.signo == "-" and "izquierdo" in primera.intervalo
    segunda = res.iteraciones[1]
    assert (segunda.a, segunda.b, segunda.p) == (1.0, 1.5, 1.25)
    assert segunda.signo == "+" and "derecho" in segunda.intervalo
    assert abs(res.raiz - 1.365230013) < 1e-4  # Burden & Faires, Ejemplo 1 (sec. 2.1)


def test_sin_cambio_de_signo():
    with pytest.raises(ValueError):
        biseccion(lambda t: t * t + 1, -1, 1)


def test_max_iter():
    res = biseccion(lambda t: t - 1 / 3, 0, 1, tol=1e-15, max_iter=5)
    assert not res.convergio
    assert len(res.iteraciones) == 5


def test_tolerancia_configurable():
    gruesa = biseccion(lambda t: t * t - 2, 1, 2, tol=1e-3)
    fina = biseccion(lambda t: t * t - 2, 1, 2, tol=1e-12)
    assert len(gruesa.iteraciones) < len(fina.iteraciones)


def test_raices_de_g_prima_ejemplo_B():
    """g'(x) = 3x^2 - 35x/2 + 49/2 tiene raíces 7/3 y 7/2; se compara con numpy.roots."""
    coef = [3, -35 / 2, 49 / 2]
    raices = buscar_raices(lambda t: np.polyval(coef, t), 2, 4, tol=1e-12)
    obtenidas = sorted(r.raiz for r in raices)
    referencia = sorted(np.roots(coef).real)
    assert len(obtenidas) == 2
    assert abs(obtenidas[0] - 7 / 3) < 1e-11 and abs(obtenidas[1] - 7 / 2) < 1e-11
    assert np.allclose(obtenidas, referencia, atol=1e-10)


def test_raices_varias_contra_numpy():
    rng = np.random.default_rng(1)
    for _ in range(10):
        nodos = np.sort(rng.uniform(-3, 3, size=rng.integers(3, 7)))
        g = np.poly(nodos)          # coeficientes de Π (x - x_i)
        dg = np.polyder(g)
        raices = buscar_raices(lambda t: np.polyval(dg, t), nodos[0], nodos[-1], subintervalos=2000)
        assert np.allclose(sorted(r.raiz for r in raices), sorted(np.roots(dg).real), atol=1e-9)


def test_cero_en_la_malla():
    intervalos, ceros = buscar_cambios_signo(lambda t: t - 0.5, 0, 1, subintervalos=10)
    assert ceros == [0.5] and intervalos == []
