"""Construcción del reporte paso a paso (texto para consola o Markdown).

Este módulo solo transforma un ``Analisis`` en texto; no hace cálculos nuevos
ni lee del teclado. Cada sección corresponde a un requisito del taller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

import sympy as sp

from .analisis import Analisis, AnalisisPunto
from .biseccion import ResultadoBiseccion
from .entrada import X
from .errores import CotaError, a_float, menor_o_igual

ANCHO = 78


# --------------------------------------------------------------------------
# Formato de números y expresiones
# --------------------------------------------------------------------------
def expr(e: sp.Expr) -> str:
    """Expresión legible: usa ^ para potencias."""
    return sp.sstr(e).replace("**", "^")


def dec(v: sp.Expr, cifras: int) -> str:
    """Decimal con ``cifras`` cifras significativas."""
    try:
        f = a_float(v)
    except (TypeError, ValueError):
        return expr(v)
    if f == 0:
        return "0"
    return f"{f:.{cifras}g}"


def num(v: sp.Expr, cifras: int, max_len: int = 40) -> str:
    """Fracción exacta y decimal (``29/88 ≈ 0.32954545``); solo el entero si es entero."""
    v = sp.sympify(v)
    if v.is_Integer:
        return str(v)
    if v.is_Rational:
        return f"{v} ≈ {dec(v, cifras)}"
    if v.is_Float:
        return dec(v, cifras)
    s = expr(v)
    if len(s) <= max_len:
        return f"{s} ≈ {dec(v, cifras)}"
    return f"≈ {dec(v, cifras)}"


def num_pct(v: sp.Expr, cifras: int) -> str:
    """Como ``num`` pero para porcentajes: ``25/22 % ≈ 1.1363636 %``."""
    v = sp.sympify(v)
    if v.is_Integer:
        return f"{v} %"
    if v.is_Rational:
        return f"{v} % ≈ {dec(v, cifras)} %"
    return f"{dec(v, cifras)} %"


def corto(v: sp.Expr, cifras: int, max_len: int = 25) -> str:
    """Valor exacto si es corto; si no, su decimal (para sustituciones en fórmulas)."""
    s = expr(v)
    return s if len(s) <= max_len else dec(v, cifras)


def par(v: sp.Expr, cifras: int = 8, fracciones: bool = True) -> str:
    """Valor entre paréntesis si no es un número simple no negativo.

    Con ``fracciones=False`` (uso en restas como ``3 - 11/4``) una fracción
    positiva no se encierra entre paréntesis.
    """
    s = corto(v, cifras)
    if re.fullmatch(r"\d+(\.\d+)?(e[+-]?\d+)?", s):
        return s
    if not fracciones and re.fullmatch(r"\d+/\d+", s):
        return s
    return f"({s})"


def tabla(encabezados: list[str], filas: list[list[str]], sangria: str = "  ") -> list[str]:
    """Tabla de texto alineada."""
    anchos = [len(h) for h in encabezados]
    for fila in filas:
        for i, celda in enumerate(fila):
            anchos[i] = max(anchos[i], len(celda))

    def linea(celdas: list[str]) -> str:
        return sangria + " | ".join(c.ljust(w) for c, w in zip(celdas, anchos)).rstrip()

    separador = sangria + "-+-".join("-" * w for w in anchos)
    return [linea(encabezados), separador, *[linea(f) for f in filas]]


def ok(valor: bool) -> str:
    return "OK" if valor else "FALLA"


# --------------------------------------------------------------------------
# Estructura del reporte
# --------------------------------------------------------------------------
@dataclass
class Seccion:
    titulo: str
    lineas: list[str] = field(default_factory=list)


def _nombre_p(n: int) -> str:
    return f"P_{n}"


def _nombre_l(n: int, k: int | str) -> str:
    return f"L_{{{n},{k}}}"


def _deriv(orden: int) -> str:
    return f"f^({orden})"


# --------------------------------------------------------------------------
# Secciones 1 a 4 (comunes a todos los puntos)
# --------------------------------------------------------------------------
def seccion_datos(an: Analisis) -> Seccion:
    c = an.config.cifras
    n = an.lagrange.n
    s = Seccion("1. Datos")
    if an.modo_datos:
        s.lineas.append("Modo: tabla de datos (sin función f(x)).")
        col = "y_k"
    else:
        s.lineas.append("Modo: función f(x) evaluada en los nodos.")
        s.lineas.append(f"f(x) = {expr(an.funcion)}")
        col = "f(x_k)"
    s.lineas.append("")
    s.lineas += tabla(["k", "x_k", col],
                      [[str(k), num(xk, c), num(fk, c)] for k, (xk, fk) in enumerate(zip(an.nodos, an.valores))])
    s.lineas.append("")
    s.lineas.append(f"Número de puntos: n + 1 = {n + 1}   =>   n = {n}")
    s.lineas.append(f"Grado del polinomio de Lagrange: a lo sumo n = {n}")
    if an.lagrange.grado < n:
        s.lineas.append(f"(Con estos datos el polinomio resultante tiene grado {an.lagrange.grado} < {n}.)")
    for aviso in an.advertencias:
        s.lineas.append("")
        s.lineas.append(f"ADVERTENCIA: {aviso}")
    return s


def seccion_bases(an: Analisis) -> Seccion:
    c = an.config.cifras
    n = an.lagrange.n
    s = Seccion("2. Polinomios base de Lagrange")
    s.lineas.append(f"{_nombre_l(n, 'k')}(x) = Π_{{i≠k}} (x - x_i) / (x_k - x_i)")
    for b in an.lagrange.bases:
        num_factores = "".join(f"({expr(X - xi)})" for xi in b.otros_nodos)
        den_factores = "".join(f"({corto(b.xk, c)} - {par(xi, c, False)})" for xi in b.otros_nodos)
        den_valores = "".join(f"({corto(d, c)})" for d in b.factores_denominador)
        s.lineas.append("")
        s.lineas.append(f"k = {b.k}:")
        s.lineas.append(f"  Numerador:   {num_factores} = {expr(b.numerador)}")
        s.lineas.append(f"  Denominador: {den_factores} = {den_valores} = {num(b.denominador, c)}")
        s.lineas.append(f"  {_nombre_l(n, b.k)}(x) = [{expr(b.numerador)}] / ({corto(b.denominador, c)})")
        s.lineas.append(f"  {_nombre_l(n, b.k)}(x) = {expr(b.expandido)}")
    return s


def seccion_polinomio(an: Analisis) -> Seccion:
    c = an.config.cifras
    lg = an.lagrange
    n = lg.n
    p = _nombre_p(n)
    s = Seccion("3. Polinomio de Lagrange")
    terminos = " + ".join(f"f(x_{k})·{_nombre_l(n, k)}(x)" for k in range(n + 1))
    s.lineas.append(f"{p}(x) = Σ_{{k=0}}^{{{n}}} f(x_k)·{_nombre_l(n, 'k')}(x) = {terminos}")
    s.lineas.append("")
    s.lineas.append("Forma sin expandir:")
    for i, (b, fk) in enumerate(zip(lg.bases, lg.valores)):
        factores = "".join(f"({expr(X - xi)})" for xi in b.otros_nodos)
        prefijo = f"{p}(x) =   " if i == 0 else " " * (len(p) + 6) + "+ "
        s.lineas.append(f"{prefijo}({corto(fk, c)})·[{factores} / ({corto(b.denominador, c)})]")
    s.lineas.append("")
    s.lineas.append(f"Sustituyendo cada {_nombre_l(n, 'k')}(x) ya expandido:")
    for i, (b, fk) in enumerate(zip(lg.bases, lg.valores)):
        prefijo = f"{p}(x) =   " if i == 0 else " " * (len(p) + 6) + "+ "
        s.lineas.append(f"{prefijo}({corto(fk, c)})·({expr(b.expandido)})")
    s.lineas.append("")
    s.lineas.append("Expandido y simplificado:")
    s.lineas.append(f"{p}(x) = {expr(lg.polinomio)}")
    s.lineas.append("")
    s.lineas.append(f"Con coeficientes decimales ({c} cifras significativas):")
    s.lineas.append(f"{p}(x) ≈ {expr(sp.N(lg.polinomio, c))}")
    return s


def seccion_verificaciones(an: Analisis) -> Seccion:
    c = an.config.cifras
    lg = an.lagrange
    n = lg.n
    s = Seccion("4. Verificaciones")
    s.lineas.append(f"a) {_nombre_p(n)}(x_k) debe ser igual a f(x_k) en cada nodo:")
    s.lineas += tabla(["k", "x_k", f"{_nombre_p(n)}(x_k)", "f(x_k)", "Estado"],
                      [[str(v.k), num(v.xk, c), num(v.p_xk, c), num(v.fk, c), ok(v.ok)]
                       for v in lg.verificaciones])
    s.lineas.append("")
    s.lineas.append(f"b) Σ_k {_nombre_l(n, 'k')}(x) = {expr(lg.suma_bases)}  (debe ser 1)   ->  {ok(lg.suma_bases_ok)}")
    return s


# --------------------------------------------------------------------------
# Secciones 5 a 9 (para cada punto x*)
# --------------------------------------------------------------------------
def seccion_valor(an: Analisis, pt: AnalisisPunto) -> Seccion:
    c = an.config.cifras
    n = an.lagrange.n
    p = _nombre_p(n)
    xs = pt.xstar
    s = Seccion(f"5. Interpolación en x* = {corto(xs, c)} (con el polinomio)")
    if pt.extrapolacion:
        s.lineas.append(
            f"ADVERTENCIA: x* = {corto(xs, c)} está fuera de [min x_i, max x_i]. Esto es una "
            "EXTRAPOLACIÓN; el error suele crecer rápido fuera de los nodos. El intervalo "
            f"[a, b] se amplía a [{corto(pt.cota.a, c)}, {corto(pt.cota.b, c)}]."
        )
        s.lineas.append("")
    sustitucion = expr(an.lagrange.polinomio.subs(X, sp.Symbol(f"({corto(xs, c)})")))
    if len(sustitucion) <= 120:
        s.lineas.append(f"{p}({corto(xs, c)}) = {sustitucion}")
    s.lineas.append(f"{p}({corto(xs, c)}) = {num(pt.valor_polinomio, c)}")
    return s


def seccion_sin_polinomio(an: Analisis, pt: AnalisisPunto) -> Seccion:
    c = an.config.cifras
    n = an.lagrange.n
    p = _nombre_p(n)
    xs = corto(pt.xstar, c)
    s = Seccion("6. Interpolación sin el polinomio")

    s.lineas.append("a) Evaluación directa de la fórmula de Lagrange en x* (no se construye P_n):")
    s.lineas.append(f"   {_nombre_l(n, 'k')}(x*) = Π_{{i≠k}} (x* - x_i) / (x_k - x_i)")
    for t in pt.directo.terminos:
        otros = [xi for i, xi in enumerate(an.nodos) if i != t.k]
        arriba = "".join(f"({xs} - {par(xi, c, False)})" for xi in otros)
        abajo = "".join(f"({corto(t.xk, c)} - {par(xi, c, False)})" for xi in otros)
        s.lineas.append(f"   {_nombre_l(n, t.k)}({xs}) = {arriba} / [{abajo}] = {num(t.lk, c)}")
    s.lineas.append("")
    s.lineas += tabla(["k", f"L_k(x*)", "f(x_k)", "L_k(x*)·f(x_k)"],
                      [[str(t.k), num(t.lk, c), num(t.fk, c), num(t.producto, c)] for t in pt.directo.terminos],
                      sangria="   ")
    s.lineas.append("")
    s.lineas.append(f"   Σ L_k(x*)·f(x_k) = {num(pt.directo.valor, c)}")
    s.lineas.append(f"   Σ L_k(x*) = {num(pt.directo.suma_l, c)}  (debe ser 1)  ->  {ok(pt.directo.suma_l_ok)}")

    s.lineas.append("")
    s.lineas.append("b) Método de Neville (Burden & Faires, Algoritmo 3.1):")
    s.lineas.append("   Q_{i,0} = f(x_i)")
    s.lineas.append("   Q_{i,j} = [(x* - x_{i-j})·Q_{i,j-1} - (x* - x_i)·Q_{i-1,j-1}] / (x_i - x_{i-j})")
    for paso in pt.neville.pasos:
        s.lineas.append(
            f"   Q_{{{paso.i},{paso.j}}} = [({xs} - {par(paso.x_izq, c, False)})·{par(paso.q_izq, c)} - "
            f"({xs} - {par(paso.x_der, c, False)})·{par(paso.q_arriba, c)}] / "
            f"({corto(paso.x_der, c)} - {par(paso.x_izq, c, False)}) = {num(paso.valor, c)}"
        )
    s.lineas.append("")
    encabezados = ["i", "x_i"] + [f"Q_{{i,{j}}}" for j in range(n + 1)]
    filas = []
    for i, fila in enumerate(pt.neville.tabla):
        filas.append([str(i), dec(an.nodos[i], c)] + [dec(q, c) for q in fila] + [""] * (n - i))
    s.lineas += tabla(encabezados, filas, sangria="   ")
    s.lineas.append("")
    s.lineas.append(f"   {p}({xs}) = Q_{{{n},{n}}} = {num(pt.neville.valor, c)}")

    s.lineas.append("")
    s.lineas.append("Comprobación (los tres valores deben coincidir):")
    s.lineas.append(f"   Polinomio expandido (sección 5): {num(pt.valor_polinomio, c)}")
    s.lineas.append(f"   Evaluación directa:              {num(pt.directo.valor, c)}   ->  {ok(pt.coincide_directo)}")
    s.lineas.append(f"   Método de Neville:               {num(pt.neville.valor, c)}   ->  {ok(pt.coincide_neville)}")
    return s


def seccion_errores(an: Analisis, pt: AnalisisPunto) -> Seccion:
    c = an.config.cifras
    p = _nombre_p(an.lagrange.n)
    xs = corto(pt.xstar, c)
    e = pt.error
    s = Seccion(f"7. Errores en x* = {xs}")
    if e.valor_real is None:
        s.lineas.append(e.nota)
        s.lineas.append("En el modo de tabla puede ingresar el valor real conocido en x* para obtenerlos.")
        return s
    origen = "valor real ingresado" if an.modo_datos else "valor verdadero"
    s.lineas.append(f"f({xs}) = {num(e.valor_real, c)}   ({origen})")
    s.lineas.append(f"{p}({xs}) = {num(e.aproximacion, c)}")
    s.lineas.append("")
    s.lineas.append(f"Error absoluto = |f(x*) - {p}(x*)| = |{corto(e.valor_real, c)} - {par(e.aproximacion, c, False)}|"
                    f" = {num(e.error_absoluto, c)}")
    if e.error_relativo is None:
        s.lineas.append(f"Error relativo: {e.nota}")
    else:
        s.lineas.append(f"Error relativo = |f(x*) - {p}(x*)| / |f(x*)| × 100 = "
                        f"{par(e.error_absoluto, c)} / |{corto(e.valor_real, c)}| × 100 = {num_pct(e.error_relativo, c)}")
        s.lineas.append("(El denominador es el valor VERDADERO f(x*), no la aproximación.)")
    return s


def _tabla_biseccion(r: ResultadoBiseccion, nombre: str, c: int) -> list[str]:
    if r.en_malla:
        return [f"   La raíz x = {r.raiz:.14g} coincidió exactamente con un punto del barrido "
                f"({nombre} = 0 allí); no hizo falta iterar."]
    filas = [[str(it.i), f"{it.a:.14g}", f"{it.b:.14g}", f"{it.p:.14g}", f"{it.fa:.{c}g}",
              f"{it.fp:.{c}g}", it.signo, it.intervalo] for it in r.iteraciones]
    lineas = [f"   Intervalo inicial [{r.a0:.14g}, {r.b0:.14g}]:"]
    lineas += tabla(["i", "a", "b", "p", f"{nombre}(a)", f"{nombre}(p)", "signo", "intervalo que se conserva"],
                    filas, sangria="   ")
    estado = "convergió" if r.convergio else "NO convergió (máximo de iteraciones)"
    lineas.append(f"   Raíz aproximada: x ≈ {r.raiz:.15g}  ({len(r.iteraciones)} iteraciones, {estado})")
    return lineas


def seccion_cota(an: Analisis, pt: AnalisisPunto) -> Seccion:
    c = an.config.cifras
    cota: CotaError = pt.cota
    n = an.lagrange.n
    o = cota.orden
    s = Seccion("8. Cota teórica del error")
    s.lineas.append(f"Teorema 3.3: f(x) - P_n(x) = f^(n+1)(ξ)/(n+1)! · Π_{{i=0}}^{{n}} (x - x_i),  ξ en [a, b]")
    s.lineas.append(f"Intervalo [a, b] = [{corto(cota.a, c)}, {corto(cota.b, c)}]  (contiene los nodos y x*)")
    if cota.extrapolacion:
        s.lineas.append("ADVERTENCIA: x* está fuera de los nodos (extrapolación); el intervalo se amplió.")
    s.lineas.append(f"Orden de la derivada: n + 1 = {o}")
    if not cota.aplicable:
        if cota.derivada is not None:
            s.lineas.append(f"{_deriv(o)}(x) = {expr(cota.derivada)}")
        s.lineas.append("")
        s.lineas.append(f"LA COTA NO SE PUEDE CALCULAR: {cota.motivo}")
        return s

    # --- M ---
    s.lineas.append("")
    s.lineas.append(f"Paso 1. M = max |{_deriv(o)}(x)| en [a, b]")
    s.lineas.append(f"   {_deriv(o)}(x) = {expr(cota.derivada)}")
    s.lineas.append(f"   {_deriv(o + 1)}(x) = {expr(cota.derivada_siguiente)}")
    if cota.derivada != 0 and cota.derivada_siguiente != 0:
        s.lineas.append(f"   Candidatos: extremos a, b y raíces de {_deriv(o + 1)} en [a, b] (bisección,"
                        f" tol = {an.config.tol:g}, barrido en {an.config.subintervalos} subintervalos).")
        if not cota.biseccion_M:
            s.lineas.append(f"   {_deriv(o + 1)} no cambia de signo en [a, b]: no hay raíces interiores.")
        for k, r in enumerate(cota.biseccion_M, 1):
            s.lineas.append(f"   Raíz {k} de {_deriv(o + 1)}:")
            s.lineas += _tabla_biseccion(r, _deriv(o + 1), c)
    if cota.nota_M:
        s.lineas.append(f"   Nota: {cota.nota_M}")
    s.lineas.append("")
    s.lineas += tabla(["x", f"{_deriv(o)}(x)", f"|{_deriv(o)}(x)|", "origen"],
                      [[num(k.x, c), num(k.valor, c), num(k.absoluto, c), k.origen] for k in cota.candidatos_M],
                      sangria="   ")
    s.lineas.append(f"   (También se revisó un muestreo denso de {an.config.muestras} puntos.)")
    donde = f" en x = {num(cota.x_M, c)}" if cota.x_M is not None else " (en todo [a, b])"
    s.lineas.append(f"   M = {num(cota.M, c)}{donde}")

    # --- factorial ---
    s.lineas.append("")
    s.lineas.append(f"Paso 2. (n+1)! = {o}! = {cota.factorial}   =>   M/(n+1)! = {num(cota.factor, c)}")

    # --- g ---
    s.lineas.append("")
    s.lineas.append("Paso 3. max |g(x)| en [a, b], con g(x) = Π_{i=0}^{n} (x - x_i)")
    factores = "".join(f"({expr(X - xi)})" for xi in an.nodos)
    s.lineas.append(f"   g(x) = {factores}")
    s.lineas.append(f"        = {expr(cota.g)}")
    s.lineas.append(f"   g'(x) = {expr(cota.dg)}")
    s.lineas.append(f"   Candidatos: extremos a, b y raíces de g'(x) en [a, b]. Se recorre [a, b] en "
                    f"{an.config.subintervalos} subintervalos buscando cambios de signo de g' y se aplica "
                    f"bisección (tol = {an.config.tol:g}, máx. {an.config.max_iter} iteraciones) en cada uno.")
    if not cota.biseccion_g:
        s.lineas.append("   g' no cambia de signo en [a, b].")
    for k, r in enumerate(cota.biseccion_g, 1):
        s.lineas.append("")
        s.lineas.append(f"   Raíz {k} de g'(x):")
        s.lineas += _tabla_biseccion(r, "g'", c)
    s.lineas.append("")
    s.lineas += tabla(["x", "g(x)", "|g(x)|", "origen"],
                      [[num(k.x, c), num(k.valor, c), num(k.absoluto, c), k.origen] for k in cota.candidatos_g],
                      sangria="   ")
    s.lineas.append("   (Las raíces se muestran como fracción cuando el valor de la bisección coincide "
                    "con un racional que anula g' exactamente.)")
    s.lineas.append(f"   max |g(x)| = {num(cota.max_g, c)} en x = {num(cota.x_max_g, c)}")

    # --- cotas ---
    xs = corto(pt.xstar, c)
    s.lineas.append("")
    s.lineas.append("Paso 4. Cotas")
    s.lineas.append(f"   Cota global:  M/(n+1)! · max|g| = {par(cota.factor, c)}·{par(cota.max_g, c)} = "
                    f"{num(cota.cota_global, c)}")
    g_fact = "".join(f"({xs} - {par(xi, c, False)})" for xi in an.nodos)
    s.lineas.append(f"   g(x*) = g({xs}) = {g_fact} = {num(pt.g_xstar, c)}")
    s.lineas.append(f"   Cota puntual: M/(n+1)! · |g(x*)| = {par(cota.factor, c)}·{par(sp.Abs(pt.g_xstar), c)} = "
                    f"{num(pt.cota_puntual, c)}")
    return s


def seccion_resumen(an: Analisis, pt: AnalisisPunto) -> Seccion:
    c = an.config.cifras
    n = an.lagrange.n
    p = _nombre_p(n)
    xs = corto(pt.xstar, c)
    e = pt.error
    s = Seccion(f"9. Resumen final (x* = {xs})")
    s.lineas.append(f"Polinomio de grado n = {n}:")
    s.lineas.append(f"   {p}(x) = {expr(an.lagrange.polinomio)}")
    s.lineas.append(f"   {p}(x) ≈ {expr(sp.N(an.lagrange.polinomio, c))}")
    s.lineas.append("")
    filas = [
        [f"{p}(x*)", num(pt.valor_polinomio, c)],
        ["   evaluación directa / Neville", f"{ok(pt.coincide_directo)} / {ok(pt.coincide_neville)}"],
        ["f(x*)", num(e.valor_real, c) if e.valor_real is not None else "desconocido"],
        ["Error absoluto", num(e.error_absoluto, c) if e.error_absoluto is not None else "no disponible"],
        ["Error relativo", num_pct(e.error_relativo, c) if e.error_relativo is not None else "no disponible"],
        ["Cota puntual del error", num(pt.cota_puntual, c) if pt.cota_puntual is not None else "no aplica"],
        ["Cota global del error", num(pt.cota.cota_global, c) if pt.cota.cota_global is not None else "no aplica"],
    ]
    if pt.extrapolacion:
        filas.append(["Advertencia", "x* fuera de los nodos (extrapolación)"])
    s.lineas += tabla(["Cantidad", "Valor"], filas, sangria="   ")
    s.lineas.append("")
    s.lineas.append("Verificación: error real ≤ cota puntual ≤ cota global")
    if e.error_absoluto is not None and pt.cota_puntual is not None:
        c1 = menor_o_igual(e.error_absoluto, pt.cota_puntual)
        c2 = menor_o_igual(pt.cota_puntual, pt.cota.cota_global)
        s.lineas.append(f"   {corto(e.error_absoluto, c)} ≤ {corto(pt.cota_puntual, c)} ≤ "
                        f"{corto(pt.cota.cota_global, c)}")
        s.lineas.append(f"   {dec(e.error_absoluto, c)} ≤ {dec(pt.cota_puntual, c)} ≤ {dec(pt.cota.cota_global, c)}"
                        f"   ->  {'se cumple (OK)' if c1 and c2 else 'NO se cumple (FALLA)'}")
    elif pt.cota_puntual is None:
        s.lineas.append("   No se puede verificar: la cota teórica no está disponible "
                        f"({'falta f(x)' if an.modo_datos else 'no aplica en este intervalo'}).")
    else:
        s.lineas.append("   No se puede verificar: falta el valor real f(x*).")
    return s


# --------------------------------------------------------------------------
# Ensamblado
# --------------------------------------------------------------------------
def generar_secciones(an: Analisis, solo_resumen: bool = False) -> list[Seccion]:
    secciones: list[Seccion] = []
    if not solo_resumen:
        secciones += [seccion_datos(an), seccion_bases(an), seccion_polinomio(an), seccion_verificaciones(an)]
    if not an.puntos and solo_resumen:
        secciones.append(seccion_polinomio(an))
    for pt in an.puntos:
        if not solo_resumen:
            secciones += [seccion_valor(an, pt), seccion_sin_polinomio(an, pt),
                          seccion_errores(an, pt), seccion_cota(an, pt)]
        secciones.append(seccion_resumen(an, pt))
    return secciones


def _encabezado(an: Analisis) -> list[str]:
    datos = f"f(x) = {expr(an.funcion)}" if an.funcion is not None else "tabla de datos"
    nodos = ", ".join(corto(x, an.config.cifras) for x in an.nodos)
    puntos = ", ".join(corto(p.xstar, an.config.cifras) for p in an.puntos) or "(ninguno)"
    return [f"Datos: {datos} | nodos: {nodos} | x*: {puntos}",
            f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}"]


def reporte_texto(an: Analisis, solo_resumen: bool = False) -> str:
    """Reporte para la consola."""
    salida = ["=" * ANCHO, "POLINOMIO DE INTERPOLACIÓN DE LAGRANGE - REPORTE PASO A PASO".center(ANCHO),
              "=" * ANCHO, *_encabezado(an)]
    for sec in generar_secciones(an, solo_resumen):
        salida += ["", "-" * ANCHO, sec.titulo, "-" * ANCHO, *sec.lineas]
    return "\n".join(salida) + "\n"


def reporte_markdown(an: Analisis, solo_resumen: bool = False) -> str:
    """Reporte en Markdown (cada sección en un bloque de texto para conservar la alineación)."""
    salida = ["# Polinomio de interpolación de Lagrange - reporte paso a paso", ""]
    salida += [f"- {linea}" for linea in _encabezado(an)]
    for sec in generar_secciones(an, solo_resumen):
        salida += ["", f"## {sec.titulo}", "", "```text", *sec.lineas, "```"]
    return "\n".join(salida) + "\n"
