"""Reemplazo mínimo de gymnasium (Env + spaces.Box), solo para usar el adaptador dentro del
contenedor de ROS sin instalar nada. El entrenamiento usa gymnasium de verdad."""
import numpy as np


class Env:
    def __init__(self, *a, **k):
        pass

    def reset(self, seed=None, options=None):
        return None


class _Box:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.low, self.high, self.shape, self.dtype = low, high, shape, dtype


class spaces:  # noqa: N801
    Box = _Box
