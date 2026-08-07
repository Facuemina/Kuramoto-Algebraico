# -*- coding: utf-8 -*-
"""
run_simulations.py

Corre todas las simulaciones numéricas de la monografía y guarda los
resultados en `data/`, para que los scripts `figura*.py` solo tengan que
cargar los datos y graficar (sin recomputar nada pesado cada vez).

Genera:
- data/sweep_kappa.npz      -> barrido en kappa (usado en Figuras 1 y 2, panel inferior)
- data/temporal_kappa05.npz -> evolución temporal a kappa fijo (usado en Figuras 1, 2 y 4)

Uso:
    python scripts/run_simulations.py
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src import config as cfg
from src.functions import integracion_rk4, kuramoto_deriv, complex_kuramoto_step, order_parameter
from src.network import build_ring_network


def run_sweep(A, W_nd, W_d, w):
    """Barrido sobre kappa: r(kappa) para KM, dKM, modelo complejo (con y sin retardo)."""
    print("Simulando barrido en kappa (Figuras 1 y 2, panel inferior)...")

    _, _, eta, _, _ = build_ring_network(cfg.N, cfg.K_NEIGHBORS, w, cfg.V_PROP)

    R_KM, R_dKM, R_cKM, R_cdKM = [], [], [], []

    for kappa in cfg.KAPPAS:
        theta0 = np.random.uniform(-np.pi, np.pi, cfg.N)
        x0 = np.exp(1j * theta0)

        # 1. Original KM (sin retardo, eta=0)
        _, theta_KM = integracion_rk4(
            lambda t, th: kuramoto_deriv(t, th, A, w, kappa, 0),
            0, theta0, cfg.T_END_SWEEP, cfg.DT,
        )
        R_KM.append(np.mean(order_parameter(theta_KM)))

        # 2. Original dKM (con retardo)
        _, theta_dKM = integracion_rk4(
            lambda t, th: kuramoto_deriv(t, th, A, w, kappa, eta),
            0, theta0, cfg.T_END_SWEEP, cfg.DT,
        )
        R_dKM.append(np.mean(order_parameter(theta_dKM)))

        # 3. Modelo complejo (sin retardo)
        x_cKM = x0.copy()
        R_cKM_temp = []
        for _ in np.arange(0, cfg.T_END_SWEEP, cfg.ZETA):
            x_cKM = complex_kuramoto_step(x_cKM, kappa * W_nd, w, cfg.ZETA)
            R_cKM_temp.append(np.abs(np.mean(x_cKM)))
        R_cKM.append(np.mean(R_cKM_temp))

        # 4. Modelo complejo (con retardo)
        x_cdKM = x0.copy()
        R_cdKM_temp = []
        for _ in np.arange(0, cfg.T_END_SWEEP, cfg.ZETA):
            x_cdKM = complex_kuramoto_step(x_cdKM, kappa * W_d, w, cfg.ZETA)
            R_cdKM_temp.append(np.abs(np.mean(x_cdKM)))
        R_cdKM.append(np.mean(R_cdKM_temp))

    return np.array(R_KM), np.array(R_dKM), np.array(R_cKM), np.array(R_cdKM)


def run_temporal(A, W_nd, W_d, w, eta):
    """Evolución temporal a kappa fijo: patrones espaciotemporales (Figuras 1, 2 y 4)."""
    print("Simulando evolución temporal a kappa = %.2f (Figuras 1, 2 y 4)..." % cfg.KAPPA_FIXED)

    np.random.seed(cfg.SEED_TEMPORAL)
    theta0 = np.random.uniform(-np.pi, np.pi, cfg.N)

    # Original KM y dKM con RK4
    t_vals, theta_KM = integracion_rk4(
        lambda t, th: kuramoto_deriv(t, th, A, w, cfg.KAPPA_FIXED, 0),
        0, theta0, cfg.T_END_TEMPORAL, cfg.DT,
    )
    _, theta_dKM = integracion_rk4(
        lambda t, th: kuramoto_deriv(t, th, A, w, cfg.KAPPA_FIXED, eta),
        0, theta0, cfg.T_END_TEMPORAL, cfg.DT,
    )

    # Modelo complejo (con y sin retardo)
    t_c = np.arange(0, cfg.T_END_TEMPORAL + cfg.ZETA, cfg.ZETA)
    x_cdKM = np.zeros((len(t_c), cfg.N), dtype=complex)
    x_cdKM[0] = np.exp(1j * theta0)
    x_cKM = np.zeros((len(t_c), cfg.N), dtype=complex)
    x_cKM[0] = np.exp(1j * theta0)

    for i in range(len(t_c) - 1):
        x_cdKM[i + 1] = complex_kuramoto_step(x_cdKM[i], cfg.KAPPA_FIXED * W_d, w, cfg.ZETA)
        x_cKM[i + 1] = complex_kuramoto_step(x_cKM[i], cfg.KAPPA_FIXED * W_nd, w, cfg.ZETA)

    theta_cdKM = np.angle(x_cdKM)
    theta_cKM = np.angle(x_cKM)

    return t_vals, theta_KM, theta_dKM, t_c, theta_cKM, theta_cdKM


def main():
    os.makedirs(cfg.DATA_DIR, exist_ok=True)

    A, tau, eta, W_nd, W_d = build_ring_network(cfg.N, cfg.K_NEIGHBORS, cfg.W_FREQ, cfg.V_PROP)

    # --- Barrido en kappa ---
    R_KM, R_dKM, R_cKM, R_cdKM = run_sweep(A, W_nd, W_d, cfg.W_FREQ)
    np.savez(
        os.path.join(cfg.DATA_DIR, "sweep_kappa.npz"),
        kappas=cfg.KAPPAS,
        R_KM=R_KM, R_dKM=R_dKM, R_cKM=R_cKM, R_cdKM=R_cdKM,
    )

    # --- Evolución temporal a kappa fijo ---
    t_vals, theta_KM, theta_dKM, t_c, theta_cKM, theta_cdKM = run_temporal(
        A, W_nd, W_d, cfg.W_FREQ, eta
    )
    np.savez(
        os.path.join(cfg.DATA_DIR, "temporal_kappa05.npz"),
        t_vals=t_vals, theta_KM=theta_KM, theta_dKM=theta_dKM,
        t_c=t_c, theta_cKM=theta_cKM, theta_cdKM=theta_cdKM,
    )

    print("Listo. Resultados guardados en '%s/'." % cfg.DATA_DIR)


if __name__ == "__main__":
    main()
