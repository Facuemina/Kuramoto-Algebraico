# -*- coding: utf-8 -*-
"""
functions.py

Funciones núcleo utilizadas en todas las simulaciones de la monografía:
- Integrador RK4 genérico (para el modelo de Kuramoto original, no lineal).
- Derivada del modelo de Kuramoto con retardo (dKM), bajo la aproximación
  de desfase constante (phase-lag).
- Paso temporal del modelo de valores complejos (enfoque algebraico de Muller),
  incluyendo la normalización Lambda.
- Cálculo del parámetro de orden de Kuramoto.

@author: Facundo
"""

import numpy as np


def integracion_rk4(f, t0, y0, t_end, h):
    """
    Integrador Runge-Kutta de orden 4, adaptado para arreglos 1D (o N-dimensionales).

    Parameters
    ----------
    f : callable
        Función derivada f(t, y) -> dy/dt.
    t0 : float
        Tiempo inicial.
    y0 : np.ndarray
        Condición inicial.
    t_end : float
        Tiempo final de integración.
    h : float
        Paso temporal fijo.

    Returns
    -------
    t_values : np.ndarray
        Vector de tiempos.
    y_values : np.ndarray
        Trayectoria integrada, de forma (len(t_values), len(y0)).
    """
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for i in range(len(t_values) - 1):
        t = t_values[i]
        y = y_values[i]

        k1 = f(t, y)
        k2 = f(t + h / 2.0, y + (h / 2.0) * k1)
        k3 = f(t + h / 2.0, y + (h / 2.0) * k2)
        k4 = f(t + h, y + h * k3)

        y_values[i + 1] = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t_values, y_values


def kuramoto_deriv(t, theta, A, w, epsilon, eta):
    """
    Derivada del modelo de Kuramoto con retardo (dKM), usando la aproximación
    de phase-lag: theta_j(t - tau_ij) ~= theta_j(t) - w*tau_ij (= eta_ij).

    Parameters
    ----------
    t : float
        Tiempo (no usado explícitamente, requerido por la firma del integrador).
    theta : np.ndarray, shape (N,)
        Vector de fases actuales.
    A : np.ndarray, shape (N, N)
        Matriz de adyacencia.
    w : float
        Frecuencia angular intrínseca (homogénea).
    epsilon : float
        Fuerza de acoplamiento (kappa).
    eta : np.ndarray or float
        Desfase heterogéneo w*tau_ij (0 si no hay retardo).

    Returns
    -------
    np.ndarray, shape (N,)
        Derivada temporal de las fases.
    """
    # theta[None, :] es theta_j (1xN), theta[:, None] es theta_i (Nx1)
    phase_diff = theta[None, :] - theta[:, None] - eta
    return w + epsilon * np.sum(A * np.sin(phase_diff), axis=1)


def complex_kuramoto_step(x, W, w, zeta):
    """
    Evalúa la ventana temporal analítica del modelo de valores complejos
    (enfoque algebraico de Muller), aplicando luego el operador Lambda
    que normaliza el módulo de cada componente a la unidad.

    Parameters
    ----------
    x : np.ndarray (complex), shape (N,)
        Estado complejo actual.
    W : np.ndarray (complex), shape (N, N)
        Matriz de acoplamiento (con o sin operador de retardo de Hadamard).
    w : float
        Frecuencia angular intrínseca.
    zeta : float
        Paso temporal de la iteración.

    Returns
    -------
    np.ndarray (complex), shape (N,)
        Estado complejo en t + zeta, con módulo normalizado a 1.
    """
    from scipy.linalg import expm

    M_exp = expm(zeta * W)
    x_next = np.exp(1j * w * zeta) * (M_exp @ x)
    # Operador Lambda: normaliza el módulo de cada componente a 1
    return x_next / np.abs(x_next)


def order_parameter(theta):
    """
    Calcula el parámetro de orden macroscópico de Kuramoto r(t).

    Parameters
    ----------
    theta : np.ndarray, shape (T, N) o (N,)
        Fases a lo largo del tiempo (o en un instante dado).

    Returns
    -------
    np.ndarray o float
        r(t) = |<exp(i*theta)>| promediado sobre los osciladores.
    """
    return np.abs(np.mean(np.exp(1j * theta), axis=-1))
