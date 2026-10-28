"""Métodos de comparación (sin RL). Cada uno es una función env -> acción en [-1, 1]^5.

  nominal : el filtro de fábrica (d_stop 0.30, d_slow 0.70, v_max 0.4) todo el tiempo.
  minimo  : el piso duro (d_stop 0.05) todo el tiempo: alcanza todo, pero va expuesto siempre.
  reglas  : lo que haría un ingeniero a mano: relaja el frente solo al acercarse al objeto y si
            no hay nadie a menos de 1.5 m.
Todos guardan el brazo ("despejar") apenas pueden, que es lo sensato sin RL.
"""
import numpy as np

from .entorno import MiloEnv
from .escudo import Theta

NOMINAL = Theta(0.30, 0.30, 0.40, 0.40)
MINIMO = Theta(0.05, 0.05, 0.40, 0.40)


def nominal(env):
    return MiloEnv.theta_a_accion(NOMINAL, True)


def minimo(env):
    return MiloEnv.theta_a_accion(MINIMO, True)


def reglas(env):
    cerca = any(np.hypot(*pos) < 1.5 for pos, _, edad, _ in env._personas_local() if edad < 0.5)
    if env.fase in ('acercar', 'brazo') and not cerca:
        th = Theta(0.05, 0.30, 0.40, 0.40)
    else:
        th = NOMINAL
    return MiloEnv.theta_a_accion(th, True)


def reglas_ciego(env):
    """Como 'reglas', pero NUNCA sube el brazo (no despeja): el escudo tiene que cuidar solo el
    punto ciego. Sirve para la prueba de estrés de la memoria."""
    a = reglas(env).copy()
    a[4] = -1.0
    return a


def reglas_tarde(env):
    """Como 'reglas', pero sube el brazo recién 4 s después de agarrar: durante ese rato el robot
    avanza con el cono frontal tapado (lo que pasa si la política despeja tarde)."""
    a = reglas(env).copy()
    a[4] = 1.0 if env.brazo.estado == 'carga_baja' and env.brazo.t > 4.0 else -1.0
    return a


BASELINES = {'nominal': nominal, 'minimo': minimo, 'reglas': reglas}
# Solo para la prueba de estrés (se evalúan si se piden con --solo). El sufijo _sin_memoria hace
# que evaluar.py apague la memoria del escudo.
EXTRA = {'reglas_ciego': reglas_ciego, 'reglas_ciego_sin_memoria': reglas_ciego,
         'reglas_tarde': reglas_tarde, 'reglas_tarde_sin_memoria': reglas_tarde}
BASELINES.update(EXTRA)
