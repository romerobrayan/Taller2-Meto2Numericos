"""Método de bisección propio (Burden & Faires, Algoritmo 2.1) con registro de iteraciones.

Se usa para encontrar las raíces de g'(x) (candidatos a máximo de |g|) y de
f^{(n+2)}(x) (candidatos a máximo de |f^{(n+1)}|). Para encontrar todas las
raíces de un intervalo, primero se recorre [a, b] en subintervalos pequeños
buscando cambios de signo y luego se aplica bisección en cada uno.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Funcion = Callable[[float], float]


@dataclass
class IteracionBiseccion:
    """Una fila de la tabla de bisección."""

    i: int
    a: float
    b: float
    p: float
    fa: float
    fp: float
    signo: str       # signo de f(a)·f(p): '+', '-' o '0'
    intervalo: str   # descripción del subintervalo que se conserva


@dataclass
class ResultadoBiseccion:
    raiz: float
    a0: float
    b0: float
    iteraciones: list[IteracionBiseccion] = field(default_factory=list)
    convergio: bool = True
    en_malla: bool = False  # la raíz coincidió exactamente con un punto del barrido


def _signo(v: float) -> int:
    return 0 if v == 0 else (1 if v > 0 else -1)


def biseccion(
    f: Funcion, a: float, b: float, tol: float = 1e-12, max_iter: int = 100
) -> ResultadoBiseccion:
    """Algoritmo 2.1 de Burden & Faires.

    Requiere f continua en [a, b] y f(a)·f(b) < 0. Se detiene cuando
    f(p) = 0 o (b − a)/2 < tol, o al llegar a ``max_iter`` iteraciones.
    """
    a, b = float(a), float(b)
    if a >= b:
        raise ValueError("Se requiere a < b para la bisección.")
    fa, fb = float(f(a)), float(f(b))
    if _signo(fa) * _signo(fb) > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos.")
    if fa == 0:
        return ResultadoBiseccion(a, a, b, [], True, True)
    if fb == 0:
        return ResultadoBiseccion(b, a, b, [], True, True)

    resultado = ResultadoBiseccion(raiz=math.nan, a0=a, b0=b, convergio=False)
    for i in range(1, max_iter + 1):
        p = a + (b - a) / 2
        fp = float(f(p))
        s = _signo(fa) * _signo(fp)
        if fp == 0 or (b - a) / 2 < tol:
            resultado.iteraciones.append(
                IteracionBiseccion(i, a, b, p, fa, fp, "0" if s == 0 else ("+" if s > 0 else "-"),
                                   "convergió: p es la raíz")
            )
            resultado.raiz = p
            resultado.convergio = True
            return resultado
        if s > 0:
            resultado.iteraciones.append(
                IteracionBiseccion(i, a, b, p, fa, fp, "+", "derecho [p, b]")
            )
            a, fa = p, fp
        else:
            resultado.iteraciones.append(
                IteracionBiseccion(i, a, b, p, fa, fp, "-", "izquierdo [a, p]")
            )
            b = p
    resultado.raiz = a + (b - a) / 2
    return resultado


def buscar_cambios_signo(
    f: Funcion, a: float, b: float, subintervalos: int = 199
) -> tuple[list[tuple[float, float]], list[float]]:
    """Recorre [a, b] y devuelve (subintervalos con cambio de signo, ceros exactos en la malla)."""
    malla = np.linspace(float(a), float(b), subintervalos + 1)
    valores = [float(f(t)) for t in malla]
    intervalos: list[tuple[float, float]] = []
    ceros: list[float] = []
    for k in range(subintervalos + 1):
        if not math.isfinite(valores[k]):
            continue
        if valores[k] == 0:
            ceros.append(float(malla[k]))
            continue
        if k < subintervalos and math.isfinite(valores[k + 1]) and valores[k + 1] != 0:
            if _signo(valores[k]) != _signo(valores[k + 1]):
                intervalos.append((float(malla[k]), float(malla[k + 1])))
    return intervalos, ceros


def buscar_raices(
    f: Funcion,
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 100,
    subintervalos: int = 199,
) -> list[ResultadoBiseccion]:
    """Encuentra las raíces de f en [a, b]: barrido de signos + bisección en cada cambio."""
    intervalos, ceros = buscar_cambios_signo(f, a, b, subintervalos)
    resultados = [biseccion(f, ia, ib, tol, max_iter) for ia, ib in intervalos]
    resultados += [ResultadoBiseccion(c, c, c, [], True, True) for c in ceros]
    return sorted(resultados, key=lambda r: r.raiz)
