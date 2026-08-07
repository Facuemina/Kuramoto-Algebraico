# Predicción analítica de patrones espacio-temporales en el modelo de Kuramoto con retardos temporales

Código de simulación que acompaña a la monografía sobre el enfoque algebraico
de Muller aplicado al modelo de Kuramoto con retardos temporales dependientes
de la distancia (dKM), en base al trabajo de Budzinski et al. (2023).

## Estructura del repositorio

```
kuramoto-monografia/
├── src/
│   ├── functions.py   # Integrador RK4, dinámica del dKM, paso del modelo complejo, r(t)
│   ├── network.py     # Grafo de anillo, matrices de retardo, base de Fourier discreta
│   └── config.py      # Parámetros compartidos por todas las simulaciones
├── scripts/
│   ├── run_simulations.py       # Corre las simulaciones pesadas y guarda los resultados en data/
│   ├── figura1_kuramoto.py      # Figura 1: sincronización sin retardo
│   ├── figura2_kuramoto_delay.py# Figura 2: ondas viajeras con retardo (dKM)
│   ├── figura3_autovalores.py   # Figura 3: análisis espectral (autocontenido, no depende de data/)
│   └── figura4_evolucion_modos.py # Figura 4: espectro modal (descomposición de Fourier)
├── data/       # Resultados numéricos intermedios (.npz), generados por run_simulations.py
├── Figuras/    # Figuras finales en PDF, usadas en la monografía (.tex)
└── requirements.txt
```

## Cómo reproducir las figuras

1. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Correr las simulaciones (genera los `.npz` en `data/`; puede tardar unos
   minutos dependiendo del procesador):

   ```bash
   python scripts/run_simulations.py
   ```

3. Generar cada figura por separado:

   ```bash
   python scripts/figura1_kuramoto.py
   python scripts/figura2_kuramoto_delay.py
   python scripts/figura3_autovalores.py   # no depende de run_simulations.py
   python scripts/figura4_evolucion_modos.py
   ```

Los PDFs quedan en `Figuras/`.

## Notas de reproducibilidad

- El barrido en `kappa` (Figuras 1 y 2, panel inferior) usa por defecto
  `T_END_SWEEP = 3.0 s` y 20 valores de `kappa` para que la corrida sea
  rápida. El paper original de referencia usa ventanas más largas (10 s) y
  más puntos; estos valores pueden ajustarse en `src/config.py`.
- La evolución temporal a `kappa = 0.5` (Figuras 1, 2 y 4) usa una semilla
  fija (`SEED_TEMPORAL = 23`) para que los resultados sean reproducibles.
- La Figura 3 (análisis espectral) es la única que no depende de
  `run_simulations.py`: los autovalores y autovectores se calculan
  directamente y de forma analítica a partir de la matriz de acoplamiento,
  sin necesidad de integrar el sistema.
