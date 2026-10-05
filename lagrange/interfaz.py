"""Menú interactivo en español.

Toda la lectura del teclado está aquí. Los datos inválidos se vuelven a pedir
con un mensaje claro; el programa nunca termina con un traceback.
"""

from __future__ import annotations

from typing import Callable, Sequence

import sympy as sp

from .analisis import Analisis, Configuracion, analizar
from .entrada import (
    ErrorEntrada,
    evaluar_en,
    evaluar_funcion_en_nodos,
    parsear_funcion,
    parsear_lista,
    parsear_numero,
    validar_datos,
    validar_nodos,
)
from .graficas import exportar_reporte, graficar
from .reporte import corto, expr, reporte_texto

try:  # rich es opcional: solo se usa para resaltar títulos y mensajes
    from rich.console import Console

    _CONSOLA: Console | None = Console(highlight=False)
except ImportError:  # pragma: no cover
    _CONSOLA = None


class SalirPrograma(Exception):
    """El usuario pidió salir (opción 0, Ctrl+D o Ctrl+C)."""


class Cancelado(Exception):
    """El usuario escribió 'cancelar' en una pregunta."""


AYUDA = """\
FORMATOS DE ENTRADA
  Función f(x):  1/x   sin(x)   exp(x)*cos(x)   sqrt(x)   x^3 - 2*x   log(x)
                 Funciones permitidas: sin, cos, tan, exp, log (= ln), sqrt, abs.
                 Constantes: pi, E. Use * para multiplicar (2*x, no 2x) y ^ o ** para potencias.
  Nodos:         2, 2.75, 4    o    [2, 2.75, 4]    o    2 2.75 4
                 Se aceptan fracciones (1/3) y constantes (pi/4). Los decimales se
                 convierten a fracciones exactas: 2.75 -> 11/4.
  Puntos x*:     un valor (3) o varios separados por comas (2.5, 3, 3.5).
  En cualquier pregunta puede escribir 'cancelar' para volver al menú.

QUÉ CALCULA
  - Los polinomios base L_{n,k}(x) y el polinomio de Lagrange P_n(x) (exacto).
  - P_n(x*) con el polinomio y SIN el polinomio (fórmula directa y método de Neville).
  - f(x*), error absoluto y error relativo porcentual en cada x*.
  - La cota teórica del error (global en [a, b] y puntual en x*), con las raíces
    de g'(x) halladas por bisección.
"""


class Interfaz:
    """Menú interactivo. ``leer`` y ``escribir`` se pueden sustituir en las pruebas."""

    def __init__(self, leer: Callable[[str], str] = input, escribir: Callable[[str], None] = print):
        self._leer = leer
        self._escribir = escribir
        self.config = Configuracion()
        self.funcion: sp.Expr | None = None
        self.nodos: list[sp.Expr] | None = None
        self.valores: list[sp.Expr] | None = None
        self.puntos: list[sp.Expr] = []
        self.reales: list[sp.Expr | None] = []
        self.analisis: Analisis | None = None

    # ------------------------------------------------------------------ E/S
    def escribir(self, texto: str = "", estilo: str | None = None) -> None:
        if estilo and _CONSOLA is not None and self._escribir is print:
            _CONSOLA.print(texto, style=estilo, markup=False, highlight=False)
        else:
            self._escribir(texto)

    def leer(self, mensaje: str) -> str:
        try:
            texto = self._leer(mensaje)
        except EOFError:
            raise SalirPrograma from None
        if texto.strip().lower() in ("cancelar", "c"):
            raise Cancelado
        return texto.strip()

    def pedir(self, mensaje: str, convertir: Callable[[str], object], opcional: bool = False):
        """Pregunta hasta obtener un valor válido. Con ``opcional``, Enter devuelve None."""
        while True:
            texto = self.leer(mensaje)
            if not texto:
                if opcional:
                    return None
                self.escribir("  Debe escribir un valor (o 'cancelar' para volver al menú).", "yellow")
                continue
            try:
                return convertir(texto)
            except (ErrorEntrada, ValueError, TypeError, ZeroDivisionError) as exc:
                self.escribir(f"  Error: {exc}", "red")
                self.escribir("  Intente de nuevo (escriba 'cancelar' para volver al menú).", "yellow")

    # -------------------------------------------------------------- lectores
    def _leer_funcion(self) -> sp.Expr:
        return self.pedir("f(x) = ", parsear_funcion)

    def _leer_nodos_funcion(self, f: sp.Expr) -> list[sp.Expr]:
        def convertir(texto: str) -> list[sp.Expr]:
            nodos = validar_nodos(parsear_lista(texto))
            evaluar_funcion_en_nodos(f, nodos)  # avisa si f no está definida en algún nodo
            return nodos

        return self.pedir("Nodos x0, x1, ..., xn (ej. 2, 2.75, 4): ", convertir)

    def _leer_puntos(self) -> list[sp.Expr]:
        def convertir(texto: str) -> list[sp.Expr]:
            puntos = parsear_lista(texto)
            if self.funcion is not None:
                for p in puntos:
                    evaluar_en(self.funcion, p)
            return puntos

        return self.pedir("Punto(s) x* donde interpolar (ej. 3  o  2.5, 3.5): ", convertir)

    def _leer_reales(self, puntos: Sequence[sp.Expr]) -> list[sp.Expr | None]:
        self.escribir("Si conoce el valor real de la función en x*, escríbalo para calcular los errores.")
        reales = []
        for p in puntos:
            reales.append(self.pedir(f"  Valor real en x* = {corto(p, 8)} (Enter si no lo conoce): ",
                                     parsear_numero, opcional=True))
        return reales

    # --------------------------------------------------------------- cálculo
    def recalcular(self) -> bool:
        if self.nodos is None:
            self.escribir("Primero ingrese los datos (opción 1, 2 o 9).", "yellow")
            return False
        self.escribir("Calculando...")
        self.analisis = analizar(self.nodos, self.valores, self.funcion, self.puntos,
                                 self.reales if self.funcion is None else None, self.config)
        return True

    def mostrar_reporte(self, solo_resumen: bool = False) -> None:
        if self.analisis is None and not self.recalcular():
            return
        self.escribir(reporte_texto(self.analisis, solo_resumen))

    # ---------------------------------------------------------------- opciones
    def opcion_funcion(self) -> None:
        self.escribir("\n--- Ingresar f(x) y los nodos ---")
        self.escribir("Ejemplos de f(x): 1/x, sin(x), exp(x)*cos(x), sqrt(x), x^3 - 2*x, log(x)")
        f = self._leer_funcion()
        nodos = self._leer_nodos_funcion(f)
        self.funcion, self.nodos, self.valores = f, nodos, None
        self.puntos, self.reales = [], []
        self.puntos = self._leer_puntos()
        if self.recalcular():
            self.mostrar_reporte()

    def opcion_tabla(self) -> None:
        self.escribir("\n--- Ingresar una tabla de datos (x_k, y_k) ---")
        nodos = self.pedir("Valores x0, x1, ..., xn (ej. 0, 2, 5): ",
                           lambda t: validar_nodos(parsear_lista(t)))

        def convertir_y(texto: str) -> list[sp.Expr]:
            valores = parsear_lista(texto)
            validar_datos(nodos, valores)
            return valores

        valores = self.pedir(f"Valores y0, ..., y{len(nodos) - 1} ({len(nodos)} números): ", convertir_y)
        self.funcion, self.nodos, self.valores = None, nodos, valores
        self.puntos = self._leer_puntos()
        self.reales = self._leer_reales(self.puntos)
        if self.recalcular():
            self.mostrar_reporte()

    def opcion_puntos(self) -> None:
        if self.nodos is None:
            self.escribir("Primero ingrese los datos (opción 1, 2 o 9).", "yellow")
            return
        self.puntos = self._leer_puntos()
        self.reales = self._leer_reales(self.puntos) if self.funcion is None else []
        if self.recalcular():
            self.mostrar_reporte()

    def opcion_configuracion(self) -> None:
        cfg = self.config
        self.escribir("\n--- Configuración (Enter conserva el valor actual) ---")

        def entero(minimo: int, maximo: int) -> Callable[[str], int]:
            def convertir(texto: str) -> int:
                v = int(texto)
                if not minimo <= v <= maximo:
                    raise ValueError(f"debe estar entre {minimo} y {maximo}")
                return v
            return convertir

        def positivo(texto: str) -> float:
            v = float(texto)
            if not 0 < v < 1:
                raise ValueError("debe ser un número entre 0 y 1, por ejemplo 1e-12")
            return v

        v = self.pedir(f"Cifras significativas a mostrar [{cfg.cifras}]: ", entero(1, 30), opcional=True)
        cfg.cifras = v if v is not None else cfg.cifras
        v = self.pedir(f"Tolerancia de la bisección [{cfg.tol:g}]: ", positivo, opcional=True)
        cfg.tol = v if v is not None else cfg.tol
        v = self.pedir(f"Iteraciones máximas de la bisección [{cfg.max_iter}]: ", entero(1, 10000), opcional=True)
        cfg.max_iter = v if v is not None else cfg.max_iter
        self.escribir(f"Configuración: {cfg.cifras} cifras, tol = {cfg.tol:g}, máx. {cfg.max_iter} iteraciones.")
        if self.nodos is not None:
            self.recalcular()

    def opcion_ejemplo(self) -> None:
        self.escribir("\nEjemplo de clase (Burden & Faires 3.1): f(x) = 1/x, nodos 2, 2.75, 4, x* = 3")
        self.funcion = parsear_funcion("1/x")
        self.nodos, self.valores = parsear_lista("2, 2.75, 4"), None
        self.puntos, self.reales = [sp.Integer(3)], []
        if self.recalcular():
            self.mostrar_reporte()

    def opcion_graficar(self) -> None:
        if self.nodos is None:
            self.escribir("Primero ingrese los datos (opción 1, 2 o 9).", "yellow")
            return
        if self.analisis is None and not self.recalcular():
            return
        ruta = self.pedir("Ruta de salida (Enter para salidas/grafica_<fecha>.png): ", str, opcional=True)
        ruta_salida = None if (ruta is None or not ruta.strip()) else ruta.strip()
        try:
            archivo = graficar(self.analisis, ruta=ruta_salida)
            self.escribir(f"Gráfica guardada exitosamente en: {archivo}", "green")
        except Exception as exc:
            self.escribir(f"No se pudo generar la gráfica: {exc}", "red")

    def opcion_exportar(self) -> None:
        if self.nodos is None:
            self.escribir("Primero ingrese los datos (opción 1, 2 o 9).", "yellow")
            return
        if self.analisis is None and not self.recalcular():
            return
        ruta = self.pedir("Ruta de salida (Enter para salidas/reporte_<fecha>.md): ", str, opcional=True)
        ruta_salida = None if (ruta is None or not ruta.strip()) else ruta.strip()
        try:
            archivo = exportar_reporte(self.analisis, ruta=ruta_salida)
            self.escribir(f"Reporte exportado exitosamente en: {archivo}", "green")
        except Exception as exc:
            self.escribir(f"No se pudo exportar el reporte: {exc}", "red")

    # ------------------------------------------------------------------ menú
    def _estado(self) -> str:
        if self.nodos is None:
            return "Datos actuales: (ninguno)"
        datos = f"f(x) = {expr(self.funcion)}" if self.funcion is not None else "tabla de datos"
        nodos = ", ".join(corto(x, 8) for x in self.nodos)
        puntos = ", ".join(corto(p, 8) for p in self.puntos) or "(ninguno)"
        return f"Datos actuales: {datos} | nodos: {nodos} | x*: {puntos}"

    def menu(self) -> str:
        self.escribir("")
        self.escribir("=" * 66, "cyan")
        self.escribir("  POLINOMIO DE INTERPOLACIÓN DE LAGRANGE - Métodos Numéricos", "bold cyan")
        self.escribir("=" * 66, "cyan")
        self.escribir(self._estado())
        self.escribir("""\
  1. Ingresar una función f(x) y los nodos
  2. Ingresar una tabla de datos (x, y) sin f(x)
  3. Cambiar los puntos x* a evaluar
  4. Ver el reporte completo paso a paso
  5. Ver solo el resumen final
  6. Graficar f(x) y P_n(x)
  7. Exportar el reporte a Markdown
  8. Configuración (cifras, tolerancia, iteraciones)
  9. Cargar el ejemplo de clase (f = 1/x, nodos 2, 2.75, 4, x* = 3)
 10. Ayuda: formatos de entrada
  0. Salir""")
        return self.leer("Opción: ")

    def despachar(self, opcion: str) -> None:
        acciones: dict[str, Callable[[], None]] = {
            "1": self.opcion_funcion,
            "2": self.opcion_tabla,
            "3": self.opcion_puntos,
            "4": lambda: self.mostrar_reporte(False),
            "5": lambda: self.mostrar_reporte(True),
            "6": self.opcion_graficar,
            "7": self.opcion_exportar,
            "8": self.opcion_configuracion,
            "9": self.opcion_ejemplo,
            "10": lambda: self.escribir(AYUDA),
        }
        if opcion == "0":
            raise SalirPrograma
        accion = acciones.get(opcion)
        if accion is None:
            self.escribir(f"Opción '{opcion}' no válida. Escriba un número del 0 al 10.", "yellow")
            return
        accion()

    def ejecutar(self) -> None:
        self.escribir("Bienvenido. Escriba el número de una opción y presione Enter.")
        while True:
            try:
                self.despachar(self.menu())
            except Cancelado:
                self.escribir("Operación cancelada; volviendo al menú.", "yellow")
            except (SalirPrograma, KeyboardInterrupt):
                break
            except ErrorEntrada as exc:
                self.escribir(f"Error en los datos: {exc}", "red")
            except Exception as exc:  # nunca mostrar un traceback al usuario
                self.escribir(f"Ocurrió un error inesperado ({type(exc).__name__}: {exc}). "
                              "Revise los datos e intente de nuevo.", "red")
        self.escribir("\nFin del programa. Hasta luego.")
