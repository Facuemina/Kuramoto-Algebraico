# -*- coding: utf-8 -*-
"""
network.py

Construcción de la topología de red (grafo de anillo con condiciones de
contorno periódicas), las matrices de retardo temporal dependientes de la
distancia, y la base de Fourier discreta usada en la descomposición de
Galerkin (Teorema de Diagonalización Circulante).

@author: Facundo
"""

import numpy as np


def build_ring_network(N, k_neighbors, w, v):
    """
    Construye un grafo de anillo con N nodos, donde cada nodo se conecta
    bidireccionalmente con sus k_neighbors vecinos más cercanos a cada lado,
    junto con las matrices de retardo temporal dependientes de la distancia.

    Parameters
    ----------
    N : int
        Número de nodos.
    k_neighbors : int
        Vecinos más cercanos por lado (conexiones totales por nodo = 2*k_neighbors).
    w : float
        Frecuencia angular intrínseca homogénea.
    v : float
        Velocidad de propagación de la señal (define tau = distancia / v).

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Matriz de adyacencia binaria.
    tau : np.ndarray, shape (N, N)
        Retardos temporales dependientes de la distancia métrica en el anillo.
    eta : np.ndarray, shape (N, N)
        Desfase heterogéneo equivalente eta = w * tau.
    W_nd : np.ndarray, shape (N, N)
        Matriz de acoplamiento sin retardo (W_nd = A).
    W_d : np.ndarray, shape (N, N), complex
        Matriz de acoplamiento con el operador de retardo de Hadamard aplicado
        (W_d = exp(-i*eta) o A).
    """
    X = np.arange(N).reshape(-1, 1)
    d_ij = np.abs(X - X.T)
    d_ij = np.minimum(d_ij, N - d_ij)  # distancia mínima en el anillo (periódica)

    A = np.where(d_ij <= k_neighbors, 1, 0)

    tau = d_ij / v
    eta = w * tau

    W_nd = A.astype(float)             # Sin delay
    W_d = np.exp(-1j * eta) * A        # Con delay (producto de Hadamard)

    return A, tau, eta, W_nd, W_d


def fourier_basis(N):
    """
    Construye la base de Fourier discreta (columnas = autovectores de
    cualquier matriz circulante N x N), según el Teorema de Diagonalización
    Circulante.

    Parameters
    ----------
    N : int
        Número de nodos.

    Returns
    -------
    V : np.ndarray, shape (N, N), complex
        Matriz unitaria de la DFT. La columna m es el autovector v_m del
        modo espacial m.
    """
    j = np.arange(N).reshape(-1, 1)  # índice del nodo espacial
    m = np.arange(N).reshape(1, -1)  # índice del modo de Fourier
    V = np.exp(-1j * 2 * np.pi * j * m / N) / np.sqrt(N)
    return V
