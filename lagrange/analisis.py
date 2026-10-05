"""Orquestación del cálculo completo (sin print ni input).

Une la entrada validada con los módulos de cálculo y devuelve un objeto
``Analisis`` que contiene todos los resultados; la interfaz y el reporte solo
leen este objeto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import sympy as sp

from .entrada import (
    ErrorEntrada,
    X,
    advertencias_nodos,
    evaluar_en,
    evaluar_funcion_en_nodos,
    obtener_variable,
    validar_datos,
    validar_nodos,
)
from .errores import (
    CotaError,
    ErrorPunto,
    calcular_cota,
    cota_puntual,
    error_en_punto,
)
from .interpolacion import ResultadoDirecto, ResultadoNeville, evaluacion_directa, iguales, neville
from .polinomio import ResultadoLagrange, construir_lagrange, evaluar_polinomio


@dataclass
class Configuracion:
    cifras: int = 8             # cifras significativas para mostrar decimales
    tol: float = 1e-12          # tolerancia de la bisección
    max_iter: int = 100         # iteraciones máximas de la bisección
    muestras: int = 2000        # puntos del muestreo denso (red de seguridad para M)
    subintervalos: int = 199    # subintervalos del barrido de cambios de signo


@dataclass
class AnalisisPunto:
    xstar: sp.Expr
    valor_polinomio: sp.Expr     # P_n(x*) con el polinomio expandido
    directo: ResultadoDirecto    # evaluación directa sin polinomio
    neville: ResultadoNeville    # método de Neville
    coincide_directo: bool
    coincide_neville: bool
    error: ErrorPunto
    cota: CotaError
    g_xstar: sp.Expr
    cota_puntual: sp.Expr | None

    @property
    def extrapolacion(self) -> bool:
        return self.cota.extrapolacion


@dataclass
class Analisis:
    funcion: sp.Expr | None
    nodos: list[sp.Expr]
    valores: list[sp.Expr]
    lagrange: ResultadoLagrange
    puntos: list[AnalisisPunto]
    config: Configuracion
    advertencias: list[str] = field(default_factory=list)
    var: sp.Symbol = X

    @property
    def modo_datos(self) -> bool:
        return self.funcion is None


def analizar_punto(
    funcion: sp.Expr | None,
    lagrange: ResultadoLagrange,
    xstar: sp.Expr,
    valor_real: sp.Expr | None,
    config: Configuracion,
    var: sp.Symbol = X,
) -> AnalisisPunto:
    """Interpolación, errores y cotas para un único punto x*."""
    nodos, valores = lagrange.nodos, lagrange.valores
    vp = evaluar_polinomio(lagrange.polinomio, xstar, var)
    directo = evaluacion_directa(nodos, valores, xstar)
    nev = neville(nodos, valores, xstar)

    if funcion is not None:
        valor_real = evaluar_en(funcion, xstar, var)
    error = error_en_punto(xstar, vp, valor_real)

    cota = calcular_cota(funcion, nodos, [xstar], config.tol, config.max_iter,
                         config.muestras, config.subintervalos, var=var)
    g_x, puntual = cota_puntual(cota, nodos, xstar)
    return AnalisisPunto(
        xstar=xstar,
        valor_polinomio=vp,
        directo=directo,
        neville=nev,
        coincide_directo=iguales(directo.valor, vp),
        coincide_neville=iguales(nev.valor, vp),
        error=error,
        cota=cota,
        g_xstar=g_x,
        cota_puntual=puntual,
    )


def analizar(
    nodos: Sequence[sp.Expr],
    valores: Sequence[sp.Expr] | None = None,
    funcion: sp.Expr | None = None,
    puntos: Sequence[sp.Expr] = (),
    reales: Sequence[sp.Expr | None] | None = None,
    config: Configuracion | None = None,
    var: sp.Symbol | str | None = None,
) -> Analisis:
    """Ejecuta todo el análisis.

    Modo función: se da ``funcion`` y se calculan los valores f(x_k).
    Modo datos:   se dan ``valores`` (y opcionalmente ``reales`` en cada x*).
    """
    config = config or Configuracion()
    nodos = validar_nodos(nodos)
    if funcion is None and valores is None:
        raise ErrorEntrada("Debe ingresar una función f(x) o la lista de valores y_k.")

    if var is None:
        var_sym = obtener_variable(funcion) if funcion is not None else X
    elif isinstance(var, str):
        var_sym = sp.Symbol(var.strip(), real=True)
    else:
        if funcion is not None:
            coincidentes = [s for s in funcion.free_symbols if s.name == var.name]
            var_sym = coincidentes[0] if coincidentes else sp.Symbol(var.name, real=True)
        else:
            var_sym = sp.Symbol(var.name, real=True)

    if funcion is not None:
        valores = evaluar_funcion_en_nodos(funcion, nodos, var_sym)
    else:
        validar_datos(nodos, valores)
    valores = list(valores)

    puntos = list(puntos)
    if reales is None:
        reales = [None] * len(puntos)
    if len(reales) != len(puntos):
        raise ErrorEntrada("La cantidad de valores reales no coincide con la cantidad de puntos x*.")

    lagrange = construir_lagrange(nodos, valores, var_sym)
    resultados = [analizar_punto(funcion, lagrange, p, r, config, var_sym) for p, r in zip(puntos, reales)]
    return Analisis(
        funcion=funcion,
        nodos=nodos,
        valores=valores,
        lagrange=lagrange,
        puntos=resultados,
        config=config,
        advertencias=advertencias_nodos(nodos),
        var=var_sym,
    )
