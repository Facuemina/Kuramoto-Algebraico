# -*- coding: utf-8 -*-
"""
figura4_evolucion_modos.py

Genera la Figura 4 de la monografía: "Evolución espaciotemporal del
espectro modal", obtenida al proyectar el estado del sistema (dKM, con
retardo) sobre la base de Fourier discreta (descomposición de Galerkin)
en cada instante de tiempo.

Requiere haber corrido antes `scripts/run_simulations.py`.

Uso:
    python scripts/figura4_evolucion_modos.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src import config as cfg


def main():
    os.makedirs(cfg.FIGURAS_DIR, exist_ok=True)

    temporal = np.load(os.path.join(cfg.DATA_DIR, "temporal_kappa05.npz"))
    t_vals, theta_dKM = temporal["t_vals"], temporal["theta_dKM"]

    subsample2 = np.arange(0, len(t_vals[t_vals <= cfg.T_PLOT_MAX_MODOS]), int(cfg.ZETA / cfg.DT))
    t_sub2 = t_vals[subsample2]
    n = theta_dKM.shape[1]

    # Pasamos al plano complejo para evitar el ruido de las discontinuidades 2*pi
    x = np.exp(1j * theta_dKM)

    # Descomposición de Galerkin: proyección sobre los modos de Fourier
    # espaciales (FFT sobre el eje de los nodos, normalizada por N)
    modos_temporales = np.fft.fft(x, axis=1) / n

    # Magnitud de los coeficientes de Galerkin (qué modo domina en cada instante)
    magnitud_modos = np.abs(modos_temporales)[subsample2]

    plt.figure(figsize=(10, 5))
    # Se grafica el logaritmo de la magnitud, sumando un epsilon para evitar log(0)
    plt.pcolormesh(
        np.arange(cfg.N), t_sub2, np.log(magnitud_modos + 1e-10),
        cmap="rainbow", shading="auto",
    )
    plt.colorbar(label=r"$\log_{10}|\mu_m|$")
    plt.xlabel("Modo Espacial ($m$)", fontsize=12)
    plt.ylabel("Tiempo (s)", fontsize=12)
    plt.title("Espectro Modal Espaciotemporal", fontsize=14)

    plt.tight_layout()
    out_path = os.path.join(cfg.FIGURAS_DIR, "Evolucion_de_modos.pdf")
    plt.savefig(out_path, format="pdf")
    print("Figura guardada en:", out_path)


if __name__ == "__main__":
    main()
