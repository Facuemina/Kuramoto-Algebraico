# -*- coding: utf-8 -*-
"""
figura3_autovalores.py

Genera la Figura 3 de la monografía: "Análisis geométrico y espectral del
efecto de los retardos temporales".

Panel izquierdo: autovalores de la matriz de acoplamiento en el plano
complejo, con y sin retardo temporal (Teorema de Diagonalización Circulante).
Paneles derechos: fase espacial de los autovectores dominantes (modo base
sincronizado, y el par de modos conjugados dominantes con retardo).

Este script es autocontenido: no depende de `run_simulations.py`, ya que
el análisis espectral es directo (no requiere integrar el sistema).

Uso:
    python scripts/figura3_autovalores.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src import config as cfg
from src.network import build_ring_network, fourier_basis


def plot_ring(ax, idx, eigvals, V, x_coords, y_coords, title_prefix):
    """Grafica la fase espacial del autovector `idx` sobre los nodos del anillo."""
    val = eigvals[idx]
    vec = V[:, idx]  # Autovector analítico de Fourier correspondiente a este índice

    # El argumento del autovector da la configuración de la fase espacial
    color_val = np.mod(np.angle(vec), 2 * np.pi)

    sc = ax.scatter(
        x_coords, y_coords, c=color_val, cmap="hsv", vmin=0, vmax=2 * np.pi,
        edgecolor="k", linewidth=0.5,
    )

    if "Base" in title_prefix:
        ax.set_title(f"{title_prefix} (Modo {idx + 1})\n$\\lambda = {val.real:.2f}$", fontsize=10)
    else:
        ax.set_title(
            f"{title_prefix} (Modo {idx + 1})\n$\\lambda = {val.real:.2f} {val.imag:+.2f}i$",
            fontsize=10,
        )
    ax.axis("equal")
    ax.axis("off")
    return sc


def main():
    os.makedirs(cfg.FIGURAS_DIR, exist_ok=True)

    A, tau, eta, W_nd, W_d = build_ring_network(cfg.N, cfg.K_NEIGHBORS, cfg.W_FREQ, cfg.V_PROP)
    V = fourier_basis(cfg.N)
    kappa = cfg.KAPPA_FIXED

    # --- Teorema de Diagonalización Circulante (CDT) ---
    # En matrices circulantes, los autovalores son exactamente la FFT de la
    # primera fila de la matriz.
    eigvals_nd = np.fft.fft(kappa * W_nd[0, :])
    eigvals_d = np.fft.fft(kappa * W_d[0, :])

    # --- Identificación de los modos ---
    idx_modo0 = 0  # Modo base (sincronización global) siempre es m=0

    sorted_idx = np.argsort(np.real(eigvals_d))[::-1]
    idx_top1 = sorted_idx[0]
    idx_top2 = sorted_idx[1]  # Par degenerado de la onda viajera

    idx_nd = np.argmax(np.real(eigvals_nd))  # Modo dominante sin retardo (típicamente 0)

    # --- Coordenadas espaciales de los nodos (para el layout circular) ---
    theta_nodes = np.linspace(0, 2 * np.pi, cfg.N, endpoint=False)
    x_coords = np.cos(theta_nodes)
    y_coords = np.sin(theta_nodes)

    # --- Armado de la figura ---
    fact = 0.7
    fig = plt.figure(figsize=(14 * fact, 7 * fact))
    gs = gridspec.GridSpec(2, 4, figure=fig)

    # Panel izquierdo: plano complejo
    ax_main = fig.add_subplot(gs[:, :2])
    ax_main.plot(np.real(eigvals_nd), np.imag(eigvals_nd), "bo", label="sin retardo", markerfacecolor="none")
    ax_main.plot(np.real(eigvals_nd)[idx_nd], np.imag(eigvals_nd)[idx_nd], "b*", markersize=15)

    ax_main.plot(np.real(eigvals_d), np.imag(eigvals_d), "rs", label="con retardo", markerfacecolor="none")
    ax_main.plot(np.real(eigvals_d)[idx_top1], np.imag(eigvals_d)[idx_top1], "r*", markersize=15)
    ax_main.plot(np.real(eigvals_d)[idx_top2], np.imag(eigvals_d)[idx_top2], "r*", markersize=15)

    ax_main.set_xlabel("parte real", fontsize=12)
    ax_main.set_ylabel("parte imaginaria", fontsize=12)
    ax_main.set_title("Autovalores", fontsize=14)
    ax_main.axvline(x=0, color="k", linestyle="--", linewidth=0.5)
    ax_main.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    ax_main.legend()
    ax_main.grid(True)

    # Paneles derechos: distribución de fases de los autovectores dominantes
    ax_top = fig.add_subplot(gs[0, 2:])
    sc_top = plot_ring(ax_top, idx_modo0, eigvals_nd, V, x_coords, y_coords, "Modo Base (Sincronizado)")
    cbar = plt.colorbar(sc_top, ax=ax_top, fraction=0.05, pad=0.04)
    cbar.set_label("Fase (rad)")

    ax_bot1 = fig.add_subplot(gs[1, 2])
    plot_ring(ax_bot1, idx_top1, eigvals_d, V, x_coords, y_coords, "Modo Dominante 1")

    ax_bot2 = fig.add_subplot(gs[1, 3])
    plot_ring(ax_bot2, idx_top2, eigvals_d, V, x_coords, y_coords, "Modo Dominante 2")

    plt.tight_layout()
    out_path = os.path.join(cfg.FIGURAS_DIR, "Autovalores.pdf")
    plt.savefig(out_path, format="pdf")
    print("Figura guardada en:", out_path)


if __name__ == "__main__":
    main()
