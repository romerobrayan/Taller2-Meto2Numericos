"""Interpolación en un punto x* SIN construir ni expandir el polinomio P_n.

Dos métodos independientes:

(a) Evaluación directa de la fórmula de Lagrange en x*:
        P_n(x*) = Σ_k f(x_k) · Π_{i≠k} (x* − x_i)/(x_k − x_i)
    Cada L_{n,k}(x*) es un número; nunca se forma un polinomio en x.

(b) Método de Neville (Burden & Faires, Algoritmo 3.1):
        Q_{i,0} = f(x_i)
        Q_{i,j} = [(x* − x_{i−j}) Q_{i,j−1} − (x* − x_i) Q_{i−1,j−1}] / (x_i − x_{i−j})
    y P_n(x*) = Q_{n,n}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from .entrada import validar_datos, validar_nodos


def _simp(valor: sp.Expr) -> sp.Expr:
    """Mantiene la aritmética exacta; simplifica solo valores no racionales."""
    if valor.is_Rational:
        return valor
    return sp.simplify(valor) if sp.count_ops(valor) < 200 else valor


@dataclass
class TerminoDirecto:
    """Fila k de la tabla de evaluación directa."""

    k: int
    xk: sp.Expr
    lk: sp.Expr        # L_{n,k}(x*)
    fk: sp.Expr        # f(x_k)
    producto: sp.Expr  # L_{n,k}(x*) · f(x_k)


@dataclass
class ResultadoDirecto:
    xstar: sp.Expr
    terminos: list[TerminoDirecto]
    valor: sp.Expr        # Σ L_k(x*) f(x_k)
    suma_l: sp.Expr       # Σ L_k(x*), debe ser 1

    @property
    def suma_l_ok(self) -> bool:
        return sp.simplify(self.suma_l - 1) == 0


@dataclass
class PasoNeville:
    """Cálculo detallado de un elemento Q_{i,j} con j >= 1."""

    i: int
    j: int
    x_izq: sp.Expr     # x_{i−j}
    x_der: sp.Expr     # x_i
    q_arriba: sp.Expr  # Q_{i−1,j−1}
    q_izq: sp.Expr     # Q_{i,j−1}
    valor: sp.Expr     # Q_{i,j}


@dataclass
class ResultadoNeville:
    xstar: sp.Expr
    nodos: list[sp.Expr]
    tabla: list[list[sp.Expr]]  # tabla[i][j] para j <= i
    pasos: list[PasoNeville]

    @property
    def valor(self) -> sp.Expr:
        n = len(self.nodos) - 1
        return self.tabla[n][n]


def evaluacion_directa(
    nodos: Sequence[sp.Expr], valores: Sequence[sp.Expr], xstar: sp.Expr
) -> ResultadoDirecto:
    """Calcula P_n(x*) evaluando numéricamente (y exactamente) cada L_{n,k}(x*)."""
    nodos = validar_nodos(nodos)
    validar_datos(nodos, valores)
    terminos: list[TerminoDirecto] = []
    for k, (xk, fk) in enumerate(zip(nodos, valores)):
        lk = sp.Integer(1)
        for i, xi in enumerate(nodos):
            if i != k:
                lk *= (xstar - xi) / (xk - xi)
        lk = _simp(lk)
        terminos.append(TerminoDirecto(k, xk, lk, fk, _simp(lk * fk)))
    valor = _simp(sum((t.producto for t in terminos), sp.Integer(0)))
    suma_l = _simp(sum((t.lk for t in terminos), sp.Integer(0)))
    return ResultadoDirecto(xstar, terminos, valor, suma_l)


def neville(nodos: Sequence[sp.Expr], valores: Sequence[sp.Expr], xstar: sp.Expr) -> ResultadoNeville:
    """Método de Neville: construye la tabla Q por filas (Algoritmo 3.1)."""
    nodos = validar_nodos(nodos)
    validar_datos(nodos, valores)
    n = len(nodos) - 1
    tabla: list[list[sp.Expr]] = [[sp.Integer(0)] * (i + 1) for i in range(n + 1)]
    pasos: list[PasoNeville] = []
    for i in range(n + 1):
        tabla[i][0] = valores[i]
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            x_izq, x_der = nodos[i - j], nodos[i]
            q_izq, q_arriba = tabla[i][j - 1], tabla[i - 1][j - 1]
            q = ((xstar - x_izq) * q_izq - (xstar - x_der) * q_arriba) / (x_der - x_izq)
            tabla[i][j] = _simp(q)
            pasos.append(PasoNeville(i, j, x_izq, x_der, q_arriba, q_izq, tabla[i][j]))
    return ResultadoNeville(xstar, list(nodos), tabla, pasos)


def iguales(a: sp.Expr, b: sp.Expr, tol: float = 1e-10) -> bool:
    """Compara dos resultados: exactamente si es posible, si no con tolerancia relativa."""
    diferencia = a - b
    if diferencia.is_Rational:
        return diferencia == 0
    try:
        if sp.simplify(diferencia) == 0:
            return True
    except Exception:
        pass
    da, db = float(sp.N(a, 30)), float(sp.N(b, 30))
    return abs(da - db) <= tol * max(1.0, abs(da), abs(db))
