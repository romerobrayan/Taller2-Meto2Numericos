"""Pruebas de lectura y validación de la entrada (R1, R2)."""

import pytest
import sympy as sp

from lagrange.entrada import (
    X,
    ErrorEntrada,
    advertencias_nodos,
    evaluar_funcion_en_nodos,
    parsear_funcion,
    parsear_lista,
    parsear_numero,
    validar_datos,
    validar_nodos,
)


@pytest.mark.parametrize(
    "texto",
    ["2, 2.75, 4", "[2, 2.75, 4]", "2 2.75 4", "(2; 2.75; 4)", " 2 ,2.75 , 4 "],
)
def test_formatos_de_lista(texto):
    assert parsear_lista(texto) == [2, sp.Rational(11, 4), 4]


def test_decimales_exactos():
    assert parsear_numero("2.75") == sp.Rational(11, 4)
    assert parsear_numero("0.1") == sp.Rational(1, 10)
    assert parsear_numero("-1.5") == sp.Rational(-3, 2)
    assert parsear_numero("1/3") == sp.Rational(1, 3)
    assert parsear_numero("1e-3") == sp.Rational(1, 1000)


def test_constantes_simbolicas():
    assert parsear_numero("pi/4") == sp.pi / 4
    assert parsear_lista("0, pi/4, pi/2") == [0, sp.pi / 4, sp.pi / 2]
    assert parsear_numero("sqrt(2)") == sp.sqrt(2)


@pytest.mark.parametrize("texto", ["abc", "x", "1/0", "", "2..3", "sqrt(-1)"])
def test_numeros_invalidos(texto):
    with pytest.raises(ErrorEntrada):
        parsear_numero(texto)


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("1/x", 1 / X),
        ("sin(x)", sp.sin(X)),
        ("exp(x)*cos(x)", sp.exp(X) * sp.cos(X)),
        ("sqrt(x)", sp.sqrt(X)),
        ("x^3 - 2*x", X**3 - 2 * X),
        ("log(x)", sp.log(X)),
        ("ln(x)", sp.log(X)),
        ("abs(x)", sp.Abs(X)),
        ("0.5*x", X / 2),
        ("E^x + pi", sp.exp(X) + sp.pi),
    ],
)
def test_funciones_validas(texto, esperado):
    assert sp.simplify(parsear_funcion(texto) - esperado) == 0


@pytest.mark.parametrize(
    "texto",
    [
        "__import__('os')",
        "x.__class__",
        "os.system('ls')",
        "lambda: 1",
        "y + x",
        "foo(x)",
        "sin(x",
        "x;1",
        "open('a')",
    ],
)
def test_funciones_rechazadas(texto):
    with pytest.raises(ErrorEntrada):
        parsear_funcion(texto)


def test_menos_de_dos_nodos():
    with pytest.raises(ErrorEntrada, match="al menos 2"):
        validar_nodos([sp.Integer(1)])


def test_nodos_repetidos():
    with pytest.raises(ErrorEntrada, match="repetido"):
        validar_nodos(parsear_lista("1, 2, 1.0"))


def test_longitudes_distintas():
    with pytest.raises(ErrorEntrada, match="distinto tamaño"):
        validar_datos(parsear_lista("0, 2, 5"), parsear_lista("18, 24"))


def test_funcion_no_definida_en_nodo():
    with pytest.raises(ErrorEntrada, match="no está definida"):
        evaluar_funcion_en_nodos(parsear_funcion("log(x)"), parsear_lista("0, 1, 2"))
    with pytest.raises(ErrorEntrada, match="no está definida"):
        evaluar_funcion_en_nodos(parsear_funcion("sqrt(x)"), parsear_lista("-1, 1"))


def test_valores_exactos_en_nodos():
    assert evaluar_funcion_en_nodos(parsear_funcion("1/x"), parsear_lista("2, 2.75, 4")) == [
        sp.Rational(1, 2),
        sp.Rational(4, 11),
        sp.Rational(1, 4),
    ]


def test_advertencia_runge():
    assert advertencias_nodos(list(range(11)))
    assert not advertencias_nodos(list(range(10)))
