# -*- coding: utf-8 -*-
"""
figura2_kuramoto_delay.py

Genera la Figura 2 de la monografía: "Ondas viajeras emergentes y supresión
de la sincronización global en el modelo de Kuramoto con retardo temporal
dependiente de la distancia" (dKM).

Paneles superiores: evolución espaciotemporal de las fases (modelo original
RK4 vs. modelo de valores complejos, ambos con retardo), para kappa = 0.5.
Panel inferior: parámetro de orden r en función de kappa, para ambos métodos.

Requiere haber corrido antes `scripts/run_simulations.py`.

Uso:
    python scripts/figura2_kuramoto_delay.py
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

    kappas, R_dKM, R_cdKM = sweep["kappas"], sweep["R_dKM"], sweep["R_cdKM"]
    t_vals, theta_dKM = temporal["t_vals"], temporal["theta_dKM"]
    t_c, theta_cdKM = temporal["t_c"], temporal["theta_cdKM"]

    # Mismo criterio de submuestreo que en la Figura 1, para que ambos
    # paneles superiores cubran la misma ventana temporal [0, T_PLOT_MAX]
    subsample0 = np.arange(0, len(t_vals[t_vals <= cfg.T_PLOT_MAX]), int(cfg.ZETA / cfg.DT))
    subsample = np.arange(0, len(t_c[t_c <= cfg.T_PLOT_MAX]))

    plt.figure(figsize=(8, 6))

    plt.subplot(221)
    plt.pcolormesh(
        np.arange(cfg.N), t_vals[subsample0],
        np.mod(theta_dKM[subsample0, :], 2 * np.pi),
        cmap="bone", shading="auto",
    )
    plt.title("Original, " + r"$\kappa$" + " = 0.5")
    plt.ylabel("t (s)")
    plt.xlabel("# nodo")
    plt.colorbar(label=r"$\theta$")

    plt.subplot(222)
    plt.pcolormesh(
        np.arange(cfg.N), t_c[subsample],
        np.mod(theta_cdKM[subsample, :], 2 * np.pi),
        cmap="bone", shading="auto",
    )
    plt.title("Complejo, " + r"$\kappa$" + " = 0.5")
    plt.ylabel("t (s)")
    plt.xlabel("# nodo")
    plt.colorbar(label=r"$\theta$")

    plt.subplot(212)
    plt.plot(kappas, R_dKM, "o", color="blue", label="Original")
    plt.plot(kappas, R_cdKM, ".", color="orange", label="Complejo")
    plt.xscale("log")
    plt.xlabel(r"$\kappa$")
    plt.ylabel("r")
    plt.legend()
    plt.ylim([-0.1, 1.1])

    plt.tight_layout()
    out_path = os.path.join(cfg.FIGURAS_DIR, "Kuramoto_delay_Compartido.pdf")
    plt.savefig(out_path, format="pdf")
    print("Figura guardada en:", out_path)


if __name__ == "__main__":
    main()
