"""PPO-Lagrangiano para el CMDP del paper, sobre stable-baselines3.

Problema (CMDP):
    max_pi  E[ sum_t gamma^t r_t ]
    s.a.    E[ C_exp ] <= beta_exp    (exposición: moverse con el filtro relajado)
            E[ C_soc ] <= beta_soc    (comodidad: moverse rápido cerca de personas)
donde C_* es el costo acumulado de un episodio.

Lagrangiano: L(pi, lambda) = J_r(pi) - sum_i lambda_i (J_ci(pi) - beta_i), lambda_i >= 0.
  - PPO maximiza sobre pi la recompensa penalizada r_t - lambda_exp c_exp,t - lambda_soc c_soc,t
    (el entorno la calcula, ver MiloEnv.set_lambdas).
  - Después de cada rollout, ascenso dual sobre lambda con los costos medidos:
        lambda_i <- max(0, lambda_i + eta (J_ci - beta_i))
    Si el robot se pasa del presupuesto, lambda sube (más castigo); si le sobra, baja.
"""
import csv
import json
import os
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CallbackLagrangiano(BaseCallback):
    def __init__(self, presupuestos, lambdas0=(0.1, 0.1), eta=0.05, actualizar=True,
                 ruta_csv=None, ventana=100, verbose=0):
        super().__init__(verbose)
        self.beta = np.array(presupuestos, float)
        self.lam = np.array(lambdas0, float)
        self.eta = eta
        self.actualizar = actualizar
        self.ruta_csv = ruta_csv
        self.eps_rollout = []
        self.hist = deque(maxlen=ventana)

    def _on_training_start(self):
        self.training_env.env_method('set_lambdas', *self.lam)
        if self.ruta_csv and not os.path.exists(self.ruta_csv):
            with open(self.ruta_csv, 'w', newline='') as fh:
                csv.writer(fh).writerow(['pasos', 'episodios', 'exito', 'agarro', 'choque',
                                         'viol_iso', 'exposicion', 'costo_soc', 't_ep',
                                         'lambda_exp', 'lambda_soc', 'r_tarea'])

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episodio' in info:
                self.eps_rollout.append(info['episodio'])
                self.hist.append(info['episodio'])
        return True

    def _on_rollout_end(self):
        eps = self.eps_rollout
        self.eps_rollout = []
        if not eps:
            return
        J = np.array([np.mean([e['exposicion'] for e in eps]),
                      np.mean([e['costo_soc'] for e in eps])])
        if self.actualizar:
            self.lam = np.maximum(0.0, self.lam + self.eta * (J - self.beta))
            self.training_env.env_method('set_lambdas', *self.lam)
        h = list(self.hist)
        fila = dict(
            pasos=self.num_timesteps, episodios=len(h),
            exito=np.mean([e['exito'] for e in h]), agarro=np.mean([e['agarro'] for e in h]),
            choque=np.mean([e['motivo'] == 'choque' for e in h]),
            viol_iso=np.mean([e['violaciones_iso'] for e in h]),
            exposicion=np.mean([e['exposicion'] for e in h]),
            costo_soc=np.mean([e['costo_soc'] for e in h]), t_ep=np.mean([e['t'] for e in h]),
            lambda_exp=self.lam[0], lambda_soc=self.lam[1], r_tarea=float('nan'))
        for k, v in fila.items():
            self.logger.record(f'milo/{k}', float(v))
        if self.ruta_csv:
            with open(self.ruta_csv, 'a', newline='') as fh:
                csv.writer(fh).writerow([f'{float(v):.5g}' for v in fila.values()])
        if self.verbose:
            print(f"[{self.num_timesteps:>8d}] éxito {fila['exito']:.2f}  choque "
                  f"{fila['choque']:.2f}  exp {fila['exposicion']:.2f}  soc "
                  f"{fila['costo_soc']:.2f}  lambda ({self.lam[0]:.2f}, {self.lam[1]:.2f})")

    def guardar(self, ruta):
        with open(ruta, 'w') as fh:
            json.dump(dict(lambdas=self.lam.tolist(), presupuestos=self.beta.tolist()), fh)
