"""Errores en x* y cota teórica del error de interpolación.

Teorema 3.3 (Burden & Faires):

    f(x) − P_n(x) = f^{(n+1)}(ξ)/(n+1)! · Π_{i=0}^{n} (x − x_i),   ξ ∈ [a, b]

Con g(x) = Π_{i=0}^{n} (x − x_i) y M = max_{[a,b]} |f^{(n+1)}|:

    Cota global:   |f(x) − P_n(x)| ≤ M/(n+1)! · max_{[a,b]} |g(x)|
    Cota puntual:  |f(x*) − P_n(x*)| ≤ M/(n+1)! · |g(x*)|

Los máximos se buscan entre los extremos a, b y las raíces de la derivada
(g' para |g|, f^{(n+2)} para |f^{(n+1)}|), halladas con la bisección propia.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import sympy as sp
from sympy.calculus.util import continuous_domain

from .biseccion import ResultadoBiseccion, buscar_raices
from .entrada import X

TOL_COMPARACION = 1e-12


# --------------------------------------------------------------------------
# Utilidades numéricas
# --------------------------------------------------------------------------
def a_float(v: sp.Expr) -> float:
    """Valor decimal de una expresión exacta (con precisión extendida antes de convertir)."""
    return float(sp.N(v, 30))


def funcion_numerica(expr: sp.Expr, var: sp.Symbol = X) -> Callable:
    """Versión numérica (numpy) de una expresión en var; acepta escalares o arreglos."""
    fn = sp.lambdify(var, expr, modules="numpy")

    def h(t):
        with np.errstate(all="ignore"):
            r = np.asarray(fn(t))
            if np.iscomplexobj(r):
                r = np.where(np.abs(r.imag) < 1e-14, r.real, np.nan)
            return np.broadcast_to(r.astype(float), np.shape(t)).copy()

    return h


def raiz_exacta(expr: sp.Expr, r: float, var: sp.Symbol = X) -> sp.Expr:
    """Intenta reconocer una raíz racional exacta de ``expr`` cerca de ``r``.

    Solo se acepta si al sustituirla la expresión vale exactamente 0; en caso
    contrario se devuelve el valor decimal hallado por bisección.
    """
    try:
        q = sp.nsimplify(r, rational=True, tolerance=1e-9)
        if q.is_Rational and q.q <= 10**6 and sp.simplify(expr.subs(var, q)) == 0:
            return q
    except Exception:
        pass
    return sp.Float(r, 17)


# --------------------------------------------------------------------------
# Errores en un punto (R6)
# --------------------------------------------------------------------------
@dataclass
class ErrorPunto:
    xstar: sp.Expr
    aproximacion: sp.Expr
    valor_real: sp.Expr | None
    error_absoluto: sp.Expr | None
    error_relativo: sp.Expr | None  # en porcentaje
    nota: str = ""


def error_en_punto(xstar: sp.Expr, aproximacion: sp.Expr, valor_real: sp.Expr | None) -> ErrorPunto:
    """Error absoluto |f(x*) − P_n(x*)| y relativo (%) respecto al valor VERDADERO f(x*)."""
    if valor_real is None:
        return ErrorPunto(xstar, aproximacion, None, None, None,
                          "No se conoce el valor real en x*; no se pueden calcular los errores.")
    absoluto = sp.Abs(valor_real - aproximacion)
    absoluto = absoluto if absoluto.is_Rational else sp.simplify(absoluto)
    if sp.simplify(valor_real) == 0:
        return ErrorPunto(xstar, aproximacion, valor_real, absoluto, None,
                          "f(x*) = 0: el error relativo no está definido (división por cero); "
                          "solo se reporta el error absoluto.")
    relativo = absoluto / sp.Abs(valor_real) * 100
    relativo = relativo if relativo.is_Rational else sp.simplify(relativo)
    return ErrorPunto(xstar, aproximacion, valor_real, absoluto, relativo)


# --------------------------------------------------------------------------
# Cota teórica del error (R5)
# --------------------------------------------------------------------------
@dataclass
class Candidato:
    """Punto candidato a máximo: x, valor con signo, valor absoluto y origen."""

    x: sp.Expr
    valor: sp.Expr
    absoluto: sp.Expr
    origen: str


@dataclass
class CotaError:
    aplicable: bool
    motivo: str
    a: sp.Expr
    b: sp.Expr
    extrapolacion: bool
    n: int
    orden: int = 0
    factorial: int = 1
    derivada: sp.Expr | None = None
    derivada_siguiente: sp.Expr | None = None
    M: sp.Expr | None = None
    x_M: sp.Expr | None = None
    candidatos_M: list[Candidato] = field(default_factory=list)
    biseccion_M: list[ResultadoBiseccion] = field(default_factory=list)
    nota_M: str = ""
    g: sp.Expr | None = None
    dg: sp.Expr | None = None
    biseccion_g: list[ResultadoBiseccion] = field(default_factory=list)
    candidatos_g: list[Candidato] = field(default_factory=list)
    max_g: sp.Expr | None = None
    x_max_g: sp.Expr | None = None
    factor: sp.Expr | None = None       # M/(n+1)!
    cota_global: sp.Expr | None = None
    var: sp.Symbol = X


def intervalo_trabajo(nodos: Sequence[sp.Expr], puntos: Sequence[sp.Expr]) -> tuple[sp.Expr, sp.Expr, bool]:
    """[a, b] = menor intervalo que contiene los nodos y los puntos x*; indica si hay extrapolación."""
    xmin = min(nodos, key=a_float)
    xmax = max(nodos, key=a_float)
    a, b = xmin, xmax
    extrapolacion = False
    for p in puntos:
        if a_float(p) < a_float(xmin) or a_float(p) > a_float(xmax):
            extrapolacion = True
        if a_float(p) < a_float(a):
            a = p
        if a_float(p) > a_float(b):
            b = p
    return a, b, extrapolacion


def polinomio_nodal(nodos: Sequence[sp.Expr], var: sp.Symbol = X) -> sp.Expr:
    """g(x) = Π_{i=0}^{n} (x − x_i), expandido (n+1 factores)."""
    g = sp.Integer(1)
    for xi in nodos:
        g *= var - xi
    return sp.expand(g)


def _tiene_raiz_en(arg: sp.Expr, a: sp.Expr, b: sp.Expr, var: sp.Symbol = X) -> bool:
    try:
        sol = sp.solveset(arg, var, sp.Interval(a, b))
        return sol != sp.S.EmptySet
    except Exception:
        return True


def _problema_continuidad(expr: sp.Expr, a: sp.Expr, b: sp.Expr, muestras: int, var: sp.Symbol = X) -> str | None:
    """Devuelve un mensaje si ``expr`` no es continua en [a, b]; None si lo es."""
    # Funciones con "picos" (abs, sign): no son derivables donde su argumento es 0.
    for atomo in expr.atoms(sp.Abs, sp.sign, sp.DiracDelta, sp.Heaviside):
        if _tiene_raiz_en(atomo.args[0], a, b, var):
            return f"contiene {atomo} cuyo argumento se anula en [a, b]"
    try:
        dominio = continuous_domain(expr, var, sp.Interval(a, b))
        if not sp.Interval(a, b).is_subset(dominio):
            return f"no es continua en todo [a, b] (dominio de continuidad: {dominio})"
    except Exception:
        pass  # si SymPy no puede decidirlo, se usa el muestreo
    muestra = funcion_numerica(_sin_deltas(expr), var)(np.linspace(a_float(a), a_float(b), muestras))
    if not np.all(np.isfinite(muestra)):
        return "toma valores no finitos en el muestreo de [a, b]"
    return None


def _sin_deltas(expr: sp.Expr) -> sp.Expr:
    """Quita DiracDelta (vale 0 lejos de su argumento, ya verificado)."""
    return expr.replace(lambda e: isinstance(e, sp.DiracDelta), lambda e: sp.Integer(0))


def _candidatos(
    expr: sp.Expr, a: sp.Expr, b: sp.Expr, raices: list[ResultadoBiseccion], derivada_de_raices: sp.Expr, var: sp.Symbol = X
) -> list[Candidato]:
    """Evalúa ``expr`` en a, b y en las raíces (reconocidas exactas cuando es posible)."""
    lista: list[tuple[sp.Expr, str]] = [(a, "extremo a"), (b, "extremo b")]
    for r in raices:
        xr = raiz_exacta(derivada_de_raices, r.raiz, var)
        if a_float(a) < a_float(xr) < a_float(b):
            lista.append((xr, "raíz (bisección)"))
    candidatos = []
    for xc, origen in lista:
        v = expr.subs(var, xc)
        v = v if v.is_Rational else sp.simplify(v)
        candidatos.append(Candidato(xc, v, sp.Abs(v), origen))
    return sorted(candidatos, key=lambda c: a_float(c.x))


def calcular_cota(
    f: sp.Expr | None,
    nodos: Sequence[sp.Expr],
    puntos: Sequence[sp.Expr],
    tol: float = 1e-12,
    max_iter: int = 100,
    muestras: int = 2000,
    subintervalos: int = 199,
    var: sp.Symbol = X,
) -> CotaError:
    """Calcula M, max|g| y la cota global en [a, b] (nodos + puntos x*)."""
    nodos = list(nodos)
    n = len(nodos) - 1
    a, b, extrapolacion = intervalo_trabajo(nodos, puntos)
    cota = CotaError(True, "", a, b, extrapolacion, n, orden=n + 1, factorial=math.factorial(n + 1), var=var)

    if f is None:
        cota.aplicable = False
        cota.motivo = (
            "No se puede calcular sin conocer f(x): la cota necesita la derivada "
            f"f^({n + 1})(x) y su máximo en [a, b], y una tabla de datos no la proporciona."
        )
        return cota

    # --- Continuidad de f y de f^(n+1) --------------------------------------
    problema = _problema_continuidad(f, a, b, muestras, var)
    if problema:
        cota.aplicable = False
        cota.motivo = (f"f({var}) = {f} {problema}. El teorema del error exige f continua con "
                       f"derivada de orden {n + 1} continua en [a, b]; la cota no aplica.")
        return cota

    derivada = sp.simplify(sp.diff(f, var, n + 1))
    problema = _problema_continuidad(derivada, a, b, muestras, var)
    if problema:
        cota.aplicable = False
        cota.derivada = derivada
        cota.motivo = (f"f^({n + 1})({var}) = {derivada} {problema}. El teorema del error exige "
                       f"f^({n + 1}) continua en [a, b]; la cota no aplica.")
        return cota
    derivada = _sin_deltas(derivada)
    siguiente = _sin_deltas(sp.simplify(sp.diff(derivada, var)))
    cota.derivada, cota.derivada_siguiente = derivada, siguiente

    # --- M = max |f^(n+1)| en [a, b] ----------------------------------------
    if derivada == 0:
        cota.M, cota.x_M = sp.Integer(0), None
        cota.candidatos_M = [Candidato(a, sp.Integer(0), sp.Integer(0), "extremo a"),
                             Candidato(b, sp.Integer(0), sp.Integer(0), "extremo b")]
        cota.nota_M = (f"f^({n + 1})({var}) ≡ 0: f es un polinomio de grado ≤ {n}, "
                       "por lo que P_n reproduce f exactamente.")
    else:
        if siguiente == 0:
            raices_M: list[ResultadoBiseccion] = []
            cota.nota_M = f"f^({n + 2})({var}) ≡ 0: f^({n + 1}) es constante, basta con los extremos."
        else:
            raices_M = buscar_raices(funcion_numerica(siguiente, var), a_float(a), a_float(b),
                                     tol, max_iter, subintervalos)
        cota.biseccion_M = raices_M
        cands = _candidatos(derivada, a, b, raices_M, siguiente, var)
        # Red de seguridad: muestreo denso de |f^(n+1)|.
        malla = np.linspace(a_float(a), a_float(b), muestras)
        valores = np.abs(funcion_numerica(derivada, var)(malla))
        k = int(np.argmax(valores))
        mejor = max(cands, key=lambda c: a_float(c.absoluto))
        if valores[k] > a_float(mejor.absoluto) * (1 + 1e-9) + 1e-15:
            xm = sp.Float(malla[k], 17)
            vm = derivada.subs(var, xm)
            cands.append(Candidato(xm, vm, sp.Abs(vm), "muestreo denso"))
            cota.nota_M = (f"El muestreo con {muestras} puntos encontró un valor mayor que los "
                           "candidatos analíticos; se usa ese valor.")
        cota.candidatos_M = cands
        mejor = max(cands, key=lambda c: a_float(c.absoluto))
        cota.M, cota.x_M = mejor.absoluto, mejor.x

    # --- max |g| en [a, b] ---------------------------------------------------
    g = polinomio_nodal(nodos, var)
    dg = sp.expand(sp.diff(g, var))
    cota.g, cota.dg = g, dg
    cota.biseccion_g = buscar_raices(funcion_numerica(dg, var), a_float(a), a_float(b),
                                     tol, max_iter, subintervalos)
    cota.candidatos_g = _candidatos(g, a, b, cota.biseccion_g, dg, var)
    mejor_g = max(cota.candidatos_g, key=lambda c: a_float(c.absoluto))
    cota.max_g, cota.x_max_g = mejor_g.absoluto, mejor_g.x

    cota.factor = cota.M / cota.factorial
    cota.cota_global = cota.factor * cota.max_g
    return cota


def cota_puntual(cota: CotaError, nodos: Sequence[sp.Expr], xstar: sp.Expr) -> tuple[sp.Expr, sp.Expr | None]:
    """Devuelve (g(x*), cota puntual M/(n+1)!·|g(x*)|); la cota es None si no aplica."""
    g_x = sp.Integer(1)
    for xi in nodos:
        g_x *= xstar - xi
    g_x = g_x if g_x.is_Rational else sp.simplify(g_x)
    if not cota.aplicable or cota.factor is None:
        return g_x, None
    return g_x, cota.factor * sp.Abs(g_x)


def menor_o_igual(u: sp.Expr, v: sp.Expr) -> bool:
    """u ≤ v exacto cuando es posible, con una pequeña tolerancia de punto flotante si no."""
    d = v - u
    if d.is_Rational:
        return d >= 0
    fu, fv = a_float(u), a_float(v)
    return fu <= fv + TOL_COMPARACION * max(1.0, abs(fv))
