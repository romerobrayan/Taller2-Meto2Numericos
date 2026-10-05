"""Configuración de pytest: permite importar el paquete `lagrange` desde la raíz."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
