"""Construcción simbólica y exacta de los polinomios base L_{n,k}(x) y de P_n(x).

    P_n(x) = Σ_{k=0}^{n} L_{n,k}(x) · f(x_k)
    L_{n,k}(x) = Π_{i=0, i≠k}^{n} (x − x_i) / (x_k − x_i)

(Burden & Faires, sección 3.1). Se implementa desde cero; SymPy solo se usa
para el álgebra (productos, expansión y simplificación con racionales exactos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import sympy as sp

from .entrada import X, validar_datos, validar_nodos


@dataclass
class PolinomioBase:
    """Datos de un polinomio base L_{n,k}(x)."""

    k: int
    xk: sp.Expr
    otros_nodos: list[sp.Expr]          # x_i con i ≠ k (en orden)
    factores_denominador: list[sp.Expr]  # valores (x_k − x_i)
    numerador: sp.Expr                   # Π (x − x_i) expandido
    denominador: sp.Expr                 # Π (x_k − x_i), número exacto
    expandido: sp.Expr                   # L_{n,k}(x) expandido


@dataclass
class Verificacion:
    """Comprobación P_n(x_k) = f(x_k) en un nodo."""

    k: int
    xk: sp.Expr
    p_xk: sp.Expr
    fk: sp.Expr
    ok: bool


@dataclass
class ResultadoLagrange:
    """Resultado completo de la construcción del polinomio de Lagrange."""

    nodos: list[sp.Expr]
    valores: list[sp.Expr]
    n: int
    bases: list[PolinomioBase]
    polinomio: sp.Expr                  # P_n(x) expandido y simplificado
    suma_bases: sp.Expr                 # Σ L_{n,k}(x), debe ser 1
    verificaciones: list[Verificacion] = field(default_factory=list)

    @property
    def grado(self) -> int:
        """Grado real de P_n (puede ser menor que n si los datos lo permiten)."""
        return int(sp.degree(self.polinomio, X)) if self.polinomio != 0 else 0

    @property
    def suma_bases_ok(self) -> bool:
        return sp.simplify(self.suma_bases - 1) == 0

    @property
    def verificaciones_ok(self) -> bool:
        return all(v.ok for v in self.verificaciones)


def _simplificar(expr: sp.Expr) -> sp.Expr:
    """Simplifica solo si hace falta (los racionales ya están en forma mínima)."""
    if expr.is_Rational:
        return expr
    return sp.nsimplify(expr) if expr.has(sp.Float) else sp.simplify(expr)


def polinomio_base(nodos: Sequence[sp.Expr], k: int) -> PolinomioBase:
    """Construye L_{n,k}(x) = Π_{i≠k} (x − x_i)/(x_k − x_i)."""
    xk = nodos[k]
    otros = [xi for i, xi in enumerate(nodos) if i != k]
    factores_den = [_simplificar(xk - xi) for xi in otros]

    numerador = sp.Integer(1)
    denominador = sp.Integer(1)
    for xi, d in zip(otros, factores_den):
        numerador *= X - xi
        denominador *= d
    denominador = _simplificar(denominador)
    numerador = sp.expand(numerador)
    expandido = sp.expand(numerador / denominador)
    return PolinomioBase(k, xk, otros, factores_den, numerador, denominador, expandido)


def construir_lagrange(nodos: Sequence[sp.Expr], valores: Sequence[sp.Expr]) -> ResultadoLagrange:
    """Construye todos los L_{n,k}, el polinomio P_n y sus verificaciones."""
    nodos = validar_nodos(nodos)
    validar_datos(nodos, valores)
    valores = list(valores)
    n = len(nodos) - 1

    bases = [polinomio_base(nodos, k) for k in range(n + 1)]
    polinomio = sp.expand(sum((fk * b.expandido for fk, b in zip(valores, bases)), sp.Integer(0)))
    if not all(c.is_Rational for c in sp.Poly(polinomio, X).all_coeffs()):
        polinomio = sp.expand(sp.simplify(polinomio))
    suma_bases = sp.expand(sum((b.expandido for b in bases), sp.Integer(0)))

    verificaciones = []
    for k, (xk, fk) in enumerate(zip(nodos, valores)):
        p_xk = _simplificar(polinomio.subs(X, xk))
        ok = sp.simplify(p_xk - fk) == 0
        verificaciones.append(Verificacion(k, xk, p_xk, fk, bool(ok)))

    return ResultadoLagrange(
        nodos=nodos,
        valores=valores,
        n=n,
        bases=bases,
        polinomio=polinomio,
        suma_bases=sp.simplify(suma_bases),
        verificaciones=verificaciones,
    )


def evaluar_polinomio(polinomio: sp.Expr, punto: sp.Expr) -> sp.Expr:
    """Evalúa el polinomio expandido P_n en x* de forma exacta."""
    return _simplificar(polinomio.subs(X, punto))
