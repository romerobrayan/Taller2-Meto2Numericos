"""Lectura y validación de la entrada: nodos, valores, funciones y puntos x*.

Todas las cantidades se convierten a números exactos de SymPy (por ejemplo
``"2.75"`` se convierte en ``11/4``) para que los resultados salgan como
fracciones limpias. Nunca se usa ``eval``/``exec``: las expresiones se leen con
``sympy.parse_expr`` y una lista blanca de nombres permitidos.
"""

from __future__ import annotations

import re
from typing import Sequence

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    rationalize,
    standard_transformations,
)

# Variable independiente (real, para que las derivadas de abs(x), etc. sean reales).
X = sp.Symbol("x", real=True)

# Límite a partir del cual se advierte el fenómeno de Runge.
MAX_NODOS_SIN_ADVERTENCIA = 10


class ErrorEntrada(ValueError):
    """Error de validación con un mensaje en español apto para el usuario."""


# Nombres permitidos en una función f(x).
_NOMBRES_FUNCION: dict[str, object] = {
    "x": X,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}

# Nombres permitidos en una constante (nodo, valor o x*): sin la variable x.
_NOMBRES_CONSTANTE = {k: v for k, v in _NOMBRES_FUNCION.items() if k != "x"}

# Clases que el código generado por parse_expr necesita (nada más).
_GLOBALES_PARSER: dict[str, object] = {
    "__builtins__": {},
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
    "Function": sp.Function,
}

_TRANSFORMACIONES = standard_transformations + (convert_xor, rationalize)

_RE_NUMERO_LITERAL = re.compile(r"\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?")
_RE_IDENTIFICADOR = re.compile(r"[A-Za-z_]\w*")
_RE_CARACTERES = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s]*$")
_RE_RACIONAL = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?(/[+-]?\d+)?$")


def _parsear_seguro(texto: str, permitidos: dict[str, object], que: str) -> sp.Expr:
    """Parsea ``texto`` con SymPy aceptando solo los nombres de ``permitidos``."""
    texto = texto.strip()
    if not texto:
        raise ErrorEntrada(f"La {que} está vacía.")
    if not _RE_CARACTERES.match(texto):
        raise ErrorEntrada(
            f"La {que} '{texto}' contiene caracteres no permitidos. "
            "Use números, x, + - * / ^ ( ) y las funciones permitidas."
        )
    if "__" in texto or re.search(r"\.\s*[A-Za-z_]", texto):
        raise ErrorEntrada(f"La {que} '{texto}' no es una expresión matemática válida.")

    # Revisa cada identificador contra la lista blanca (se quitan antes los
    # literales numéricos para no confundir la 'e' de 1e-3 con un nombre).
    sin_numeros = _RE_NUMERO_LITERAL.sub(" ", texto)
    for nombre in _RE_IDENTIFICADOR.findall(sin_numeros):
        if nombre not in permitidos:
            lista = ", ".join(sorted(permitidos))
            raise ErrorEntrada(
                f"Nombre no permitido '{nombre}' en la {que} '{texto}'. "
                f"Nombres permitidos: {lista}."
            )

    try:
        expr = parse_expr(
            texto,
            local_dict=dict(permitidos),
            global_dict=dict(_GLOBALES_PARSER),
            transformations=_TRANSFORMACIONES,
            evaluate=True,
        )
    except Exception as exc:  # SyntaxError, TypeError, TokenError...
        raise ErrorEntrada(f"No se pudo interpretar la {que} '{texto}' ({type(exc).__name__}).") from None

    if not isinstance(expr, sp.Expr):
        raise ErrorEntrada(f"La {que} '{texto}' no es una expresión numérica.")
    if expr.atoms(AppliedUndef):
        raise ErrorEntrada(f"La {que} '{texto}' usa funciones desconocidas.")
    return expr


def parsear_numero(texto: str) -> sp.Expr:
    """Convierte un texto en un número real exacto.

    ``"2.75"`` -> ``11/4``, ``"1/3"`` -> ``1/3``, ``"pi/4"`` -> ``pi/4``.
    """
    t = texto.strip().replace(" ", "")
    if not t:
        raise ErrorEntrada("Hay un valor vacío en la lista.")
    if _RE_RACIONAL.match(t):
        try:
            if "/" in t:
                num, den = t.split("/")
                valor = sp.Rational(num) / sp.Rational(den)
            else:
                valor = sp.Rational(t)
        except (ZeroDivisionError, ValueError, TypeError):
            raise ErrorEntrada(f"'{texto}' no es un número válido.") from None
    else:
        valor = _parsear_seguro(texto, _NOMBRES_CONSTANTE, "cantidad")
        valor = sp.nsimplify(valor) if valor.has(sp.Float) else valor
    if valor.free_symbols:
        raise ErrorEntrada(f"'{texto}' debe ser un número, no una expresión en x.")
    if valor.has(sp.zoo, sp.oo, -sp.oo, sp.nan) or valor.is_finite is False:
        raise ErrorEntrada(f"'{texto}' no es un número finito.")
    if valor.is_real is False or (valor.is_real is None and sp.im(sp.N(valor)) != 0):
        raise ErrorEntrada(f"'{texto}' no es un número real.")
    return valor


def separar_lista(texto: str) -> list[str]:
    """Divide '2, 2.75, 4', '[2, 2.75, 4]', '2;3' o '2 2.75 4' en elementos."""
    t = texto.strip()
    if t[:1] in "[(" and t[-1:] in "])":
        t = t[1:-1]
    if "," in t or ";" in t:
        partes = re.split(r"[,;]", t)
    else:
        partes = t.split()
    partes = [p.strip() for p in partes]
    if any(p == "" for p in partes):
        raise ErrorEntrada(f"La lista '{texto}' tiene elementos vacíos (revise las comas).")
    return partes


def parsear_lista(texto: str) -> list[sp.Expr]:
    """Convierte un texto con varios números en una lista de números exactos."""
    if not texto or not texto.strip():
        raise ErrorEntrada("La lista está vacía.")
    return [parsear_numero(p) for p in separar_lista(texto)]


def parsear_funcion(texto: str) -> sp.Expr:
    """Convierte un texto como ``'1/x'`` o ``'exp(x)*cos(x)'`` en una expresión de SymPy."""
    expr = _parsear_seguro(texto, _NOMBRES_FUNCION, "función")
    otros = expr.free_symbols - {X}
    if otros:
        raise ErrorEntrada(f"La función solo puede depender de x (encontrado: {otros}).")
    return expr


def validar_nodos(nodos: Sequence[sp.Expr]) -> list[sp.Expr]:
    """Verifica que haya al menos 2 nodos y que sean distintos."""
    nodos = list(nodos)
    if len(nodos) < 2:
        raise ErrorEntrada(
            f"Se necesitan al menos 2 nodos (n+1 >= 2); se ingresaron {len(nodos)}."
        )
    for i in range(len(nodos)):
        for j in range(i + 1, len(nodos)):
            if sp.simplify(nodos[i] - nodos[j]) == 0:
                raise ErrorEntrada(
                    f"Nodo repetido: x_{i} = x_{j} = {nodos[i]}. Los nodos deben ser "
                    "distintos porque el denominador (x_k - x_i) sería 0."
                )
    return nodos


def validar_datos(nodos: Sequence[sp.Expr], valores: Sequence[sp.Expr]) -> None:
    """Verifica que la tabla de datos tenga el mismo número de x que de y."""
    if len(nodos) != len(valores):
        raise ErrorEntrada(
            f"Los arreglos tienen distinto tamaño: {len(nodos)} valores de x y "
            f"{len(valores)} valores de y."
        )


def evaluar_en(f: sp.Expr, punto: sp.Expr) -> sp.Expr:
    """Evalúa f en un punto de forma exacta; lanza ErrorEntrada si no está definida."""
    try:
        valor = sp.simplify(f.subs(X, punto))
    except Exception:
        valor = sp.nan
    if (
        valor.has(sp.zoo, sp.oo, -sp.oo, sp.nan)
        or valor.free_symbols
        or valor.is_finite is False
        or valor.is_real is False
    ):
        raise ErrorEntrada(f"La función f(x) = {f} no está definida (en los reales) en x = {punto}.")
    return valor


def evaluar_funcion_en_nodos(f: sp.Expr, nodos: Sequence[sp.Expr]) -> list[sp.Expr]:
    """Calcula f(x_k) para cada nodo."""
    return [evaluar_en(f, xk) for xk in nodos]


def advertencias_nodos(nodos: Sequence[sp.Expr]) -> list[str]:
    """Devuelve advertencias no fatales sobre el conjunto de nodos."""
    avisos: list[str] = []
    if len(nodos) > MAX_NODOS_SIN_ADVERTENCIA:
        avisos.append(
            f"Se ingresaron {len(nodos)} nodos (grado {len(nodos) - 1}). Con más de "
            f"{MAX_NODOS_SIN_ADVERTENCIA} nodos el polinomio puede oscilar cerca de los "
            "extremos (fenómeno de Runge) y el cálculo se vuelve mal condicionado."
        )
    return avisos
