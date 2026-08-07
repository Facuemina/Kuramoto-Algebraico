# -*- coding: utf-8 -*-
"""
figura1_kuramoto.py

Genera la Figura 1 de la monografía: "Dinámica de sincronización y transición
de fase en el modelo de Kuramoto" (sin retardo temporal).

Paneles superiores: evolución espaciotemporal de las fases (modelo original
RK4 vs. modelo de valores complejos), para kappa = 0.5.
Panel inferior: parámetro de orden r en función de kappa, para ambos métodos.

Requiere haber corrido antes `scripts/run_simulations.py`.

Uso:
    python scripts/figura1_kuramoto.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src import config as cfg


def main():
    os.makedirs(cfg.FIGURAS_DIR, exist_ok=True)

    sweep = np.load(os.path.join(cfg.DATA_DIR, "sweep_kappa.npz"))
    temporal = np.load(os.path.join(cfg.DATA_DIR, "temporal_kappa05.npz"))

    kappas, R_KM, R_cKM = sweep["kappas"], sweep["R_KM"], sweep["R_cKM"]
    t_vals, theta_KM = temporal["t_vals"], temporal["theta_KM"]
    t_c, theta_cKM = temporal["t_c"], temporal["theta_cKM"]

    # Submuestreo para graficar solo hasta T_PLOT_MAX, igualando la resolución
    # temporal de ambos métodos (RK4 usa dt, el modelo complejo usa zeta)
    subsample0 = np.arange(0, len(t_vals[t_vals <= cfg.T_PLOT_MAX]), int(cfg.ZETA / cfg.DT))
    subsample = np.arange(0, len(t_c[t_c <= cfg.T_PLOT_MAX]))

    plt.figure(figsize=(8, 6))

    plt.subplot(221)
    plt.pcolormesh(
        np.arange(cfg.N), t_vals[subsample0],
        np.mod(theta_KM[subsample0, :], 2 * np.pi),
        cmap="bone", shading="auto",
    )
    plt.title("Original, " + r"$\kappa$" + " = 0.5")
    plt.ylabel("t (s)")
    plt.xlabel("# nodo")
    plt.colorbar(label=r"$\theta$")

    plt.subplot(222)
    plt.pcolormesh(
        np.arange(cfg.N), t_c[subsample],
        np.mod(theta_cKM[subsample, :], 2 * np.pi),
        cmap="bone", shading="auto",
    )
    plt.title("Complejo, " + r"$\kappa$" + " = 0.5")
    plt.ylabel("t (s)")
    plt.xlabel("# nodo")
    plt.colorbar(label=r"$\theta$")

    plt.subplot(212)
    plt.plot(kappas, R_KM, "o", color="blue", label="Original")
    plt.plot(kappas, R_cKM, ".", color="orange", label="Complejo")
    plt.xscale("log")
    plt.xlabel(r"$\kappa$")
    plt.ylabel("r")
    plt.legend()

    plt.tight_layout()
    out_path = os.path.join(cfg.FIGURAS_DIR, "Kuramoto_Compartido.pdf")
    plt.savefig(out_path, format="pdf")
    print("Figura guardada en:", out_path)


if __name__ == "__main__":
    main()
