"""Personas del simulador 2D y el detector de personas.

Cada persona son dos piernas (lo que corta el plano del lidar a 0.36 m) y un torso más ancho
arriba (lo que puede tocar el brazo). Pueden estar quietas o caminar entre dos puntos, ida y
vuelta, a velocidad constante y con pausas en los extremos. NO esquivan al robot (peor caso), pero
sí se detienen si lo tienen justo delante: si no, habría choques que el robot no puede evitar.

Detector: entrega la posición con ruido gaussiano y la velocidad estimada, pero SOLO si el lidar
le ve alguna pierna en este scan (como un detector de piernas real). Si la persona queda en la
sombra del brazo o detrás de un mueble, el robot no sabe dónde está.
"""
import math

import numpy as np

from . import config as C
from .geometria import Circulo


class Humano:
    def __init__(self, ident, pos, rumbo=0.0, puntos=None, v=0.0, pausa=1.0, rng=None):
        self.id = ident
        self.pos = np.array(pos, float)
        self.rumbo = rumbo
        self.puntos = [np.array(p, float) for p in puntos] if puntos else None
        self.v_nom = v
        self.pausa_max = pausa
        self.k = 1  # índice del punto al que va
        self.espera = 0.0
        self.t_bloq = 0.0
        self.vel = np.zeros(2)
        self.rng = rng or np.random.default_rng()

    @property
    def camina(self):
        return self.puntos is not None

    def formas(self):
        c, s = math.cos(self.rumbo), math.sin(self.rumbo)
        lado = np.array([-s, c]) * C.PIERNA_SEP
        return [Circulo(self.pos + lado, C.PIERNA_R, C.PIERNA_Z, 'persona', self.id),
                Circulo(self.pos - lado, C.PIERNA_R, C.PIERNA_Z, 'persona', self.id),
                Circulo(self.pos.copy(), C.TORSO_R, C.TORSO_Z, 'persona', self.id)]

    def paso(self, dt, bloqueado):
        """bloqueado(pos_siguiente) -> True si el robot está en el camino."""
        self.vel = np.zeros(2)
        if not self.camina:
            return
        if self.espera > 0:
            self.espera -= dt
            return
        meta = self.puntos[self.k]
        d = meta - self.pos
        dist = float(np.hypot(*d))
        if dist < 1e-3:
            self.k = 1 - self.k
            self.espera = self.rng.uniform(0, self.pausa_max)
            return
        u = d / dist
        paso = min(self.v_nom * dt, dist)
        if bloqueado(self.pos + u * (paso + 0.45)):
            # se queda quieta mientras el robot le tape el camino; si pasa más de 1.5 s, se da
            # vuelta (como haría una persona), así no quedan los dos esperándose para siempre
            self.t_bloq += dt
            if self.t_bloq > 1.5:
                self.k, self.t_bloq = 1 - self.k, 0.0
            return
        self.t_bloq = 0.0
        self.rumbo = math.atan2(u[1], u[0])
        self.pos = self.pos + u * paso
        self.vel = u * self.v_nom


class Detector:
    """Lo que el robot sabe de cada persona: última posición vista, velocidad y hace cuánto."""

    def __init__(self, sigma, p_perder, rng):
        self.sigma, self.p_perder, self.rng = sigma, p_perder, rng
        self.tracks = {}  # id -> dict(pos, vel, t, n)
        self.movimiento_visto = False

    def actualizar(self, t, humanos, vistos):
        """Agrega las detecciones de este scan. La velocidad sale de una recta ajustada a las
        posiciones del último segundo (con ruido de 5 cm, derivar entre dos scans daría ~0.7 m/s
        de ruido; con un segundo de datos queda en ~0.1 m/s)."""
        for h in humanos:
            if h.id not in vistos or self.rng.random() < self.p_perder:
                continue
            z = h.pos + self.rng.normal(0, self.sigma, 2)
            tr = self.tracks.setdefault(h.id, dict(hist=[], n=0))
            tr['hist'] = [(ti, zi) for ti, zi in tr['hist'] if t - ti <= 1.0] + [(t, z)]
            tr['n'] += 1
            tr['pos'], tr['t'] = z, t
            if len(tr['hist']) >= 4:
                ts = np.array([q[0] for q in tr['hist']])
                zs = np.array([q[1] for q in tr['hist']])
                tc = ts - ts.mean()
                vel = (tc[:, None] * (zs - zs.mean(axis=0))).sum(axis=0) / max((tc ** 2).sum(),
                                                                               1e-6)
                tr['vel'] = np.clip(vel, -1.5, 1.5)
                # la posición es la última medida (con su ruido, que cubre Z_d): suavizarla con la
                # recta la atrasa justo cuando la persona da la vuelta
            else:
                tr['vel'] = np.zeros(2)

    def estimaciones(self, t):
        """Lista de (pos, vel, edad, v_cota) de las personas que alguna vez vio.

        v_cota: cota de su velocidad para el escudo. Hasta tener 10 detecciones (~1 s) la velocidad
        estimada no es confiable y se usa V_HUMANO_MAX; después, la estimada + 0.3 m/s."""
        out = []
        for tr in self.tracks.values():
            v = float(np.hypot(*tr['vel']))
            cota = C.V_HUMANO_MAX if tr['n'] < 10 else min(v + 0.3, C.V_HUMANO_MAX)
            out.append((tr['pos'], tr['vel'], t - tr['t'], cota))
        return out

    def hay_movimiento(self):
        """¿Hay gente caminando en la escena? Queda en True apenas se ve a alguien moverse (una
        persona que se detuvo un momento puede volver a caminar)."""
        if not self.movimiento_visto:
            self.movimiento_visto = any(float(np.hypot(*tr['vel'])) > 0.25 and tr['n'] >= 10
                                        for tr in self.tracks.values())
        return self.movimiento_visto
