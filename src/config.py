# -*- coding: utf-8 -*-
"""
config.py

Parámetros compartidos por todas las simulaciones de la monografía.
Modificar acá para reproducir el paper completo (ver comentarios) o
para correr una versión más liviana/rápida.

@author: Facundo
"""

import numpy as np

# ---------------------------------------------------------------
# Parámetros de la red (grafo de anillo)
# ---------------------------------------------------------------
N = 100                 # Número de osciladores/nodos
K_NEIGHBORS = 25        # Vecinos más cercanos a cada lado (50 conexiones/nodo)
W_FREQ = 20 * np.pi     # Frecuencia angular intrínseca homogénea (omega)
V_PROP = 483.87         # Velocidad de propagación (determina el retardo tau = d/v)

# ---------------------------------------------------------------
# Barrido en kappa (Figuras 1 y 2, panel de r vs kappa)
# ---------------------------------------------------------------
# Nota: el paper original usa más puntos; acá se reduce a 20 por velocidad.
KAPPAS = np.logspace(-4, 1, 20)
T_END_SWEEP = 1.0       # El paper usa 10s; se reduce a 1s para una corrida rápida
DT = 0.001              # Paso de integración RK4 (modelo original)
ZETA_FACTOR = 5         # zeta = DT * ZETA_FACTOR (paso del modelo complejo)
ZETA = DT * ZETA_FACTOR

# ---------------------------------------------------------------
# Evolución temporal a kappa fijo (Figuras 1, 2 y 4: patrones espaciotemporales)
# ---------------------------------------------------------------
KAPPA_FIXED = 0.5
T_END_TEMPORAL = 10.0
SEED_TEMPORAL = 23       # Semilla para reproducibilidad de la condición inicial
T_PLOT_MAX = 3.0         # Ventana temporal graficada en los paneles superiores (Fig. 1 y 2)
T_PLOT_MAX_MODOS = 6.0   # Ventana temporal graficada en el heatmap de modos (Fig. 4)

# ---------------------------------------------------------------
# Rutas de salida
# ---------------------------------------------------------------
DATA_DIR = "data"
FIGURAS_DIR = "Figuras"
