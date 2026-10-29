"""Modelo 2D del brazo de Milo: alcance, sus dos "sombras" y la máquina de estados.

El brazo real (6 GDL + lift) se resume en una CÁPSULA que va del hombro (eje de joint_1) a la
pinza, con un rango de alturas. De ahí salen las dos sombras del paper:
  - sombra de SENSADO: si la cápsula corta el plano del lidar (0.363 m), tapa los rayos que la
    cruzan (el lidar la "ve" como un obstáculo propio y lo de atrás queda oculto).
  - sombra de CHOQUE: la parte de la cápsula fuera de la base se suma a la huella del robot:
    F(q) = F_base ∪ cápsula(q). Con el brazo afuera el robot es más largo y gira más ancho.

Estados (y cuánto duran):
  guardado --(alcanzar, 2 s)--> alcanzando --> agarrando (1 s) --> retrayendo (1 s) --> carga_baja
  carga_baja --(acción "despejar", 2 s, la base se queda quieta)--> despejando --> guardado
"""
import math

import numpy as np

from . import config as C
from .geometria import Capsula, a_mundo


def radio_horizontal(z):
    """Cuánto estira el brazo en horizontal (desde el eje del hombro) para llegar a la altura z."""
    z0, z1 = C.HOMBRO_Z
    dz = (z0 - z) if z < z0 else (z - z1) if z > z1 else 0.0
    return math.sqrt(max(C.RADIO_BRAZO ** 2 - dz ** 2, 0.0))


def margen_alcance(p_local, z):
    """> 0 si el objeto (en base_footprint) está al alcance. Es lo que le falta / le sobra [m]."""
    d = math.hypot(p_local[0] - C.HOMBRO_X, p_local[1])
    return radio_horizontal(z) - C.MARGEN_ALCANCE - d


def limites_profundidad(z, d_stop_nom=C.D_STOP_NOMINAL, d_min=0.05):
    """Hasta qué profundidad (dentro del mueble, desde el borde) se alcanza un objeto a la altura z
    con el filtro nominal y con el piso duro. El robot llega de frente con el objeto adelante."""
    alc = C.HOMBRO_X + radio_horizontal(z) - C.MARGEN_ALCANCE - C.FP_X_MAX
    return alc - d_stop_nom, alc - d_min


class Brazo:
    def __init__(self):
        self.reset()

    def reset(self):
        self.estado = 'guardado'
        self.t = 0.0
        self.objetivo = None
        self.tcp = np.array(C.GUARDADO_TCP, float)
        self.zr = (0.90, 1.30)
        self.con_objeto = False

    def alcanzar(self, p_local, z):
        self.estado, self.t = 'alcanzando', 0.0
        self.objetivo = (np.array(p_local, float), float(z))

    def despejar(self):
        if self.estado == 'carga_baja':
            self.estado, self.t = 'despejando', 0.0
            return True
        return False

    def paso(self, dt):
        """Avanza el brazo. Devuelve un evento: 'agarro' cuando cierra la pinza, si no None."""
        self.t += dt
        ev = None
        if self.estado == 'alcanzando':
            f = min(self.t / C.T_ALCANZAR, 1.0)
            p, z = self.objetivo
            self.tcp = (1 - f) * np.array(C.GUARDADO_TCP) + f * p
            self.zr = _interp((0.90, 1.30), _banda_alcance(z), f)
            if f >= 1.0:
                self.estado, self.t = 'agarrando', 0.0
        elif self.estado == 'agarrando':
            if self.t >= C.T_AGARRAR:
                self.estado, self.t, self.con_objeto = 'retrayendo', 0.0, True
                self._desde = (self.tcp.copy(), self.zr)
                ev = 'agarro'
        elif self.estado == 'retrayendo':
            f = min(self.t / 1.0, 1.0)
            p0, z0 = self._desde
            self.tcp = (1 - f) * p0 + f * np.array(C.CARGA_BAJA_TCP)
            self.zr = _interp(z0, C.CARGA_BAJA_Z, f)
            if f >= 1.0:
                self.estado, self.t = 'carga_baja', 0.0
        elif self.estado == 'despejando':
            f = min(self.t / C.T_DESPEJAR, 1.0)
            self.tcp = (1 - f) * np.array(C.CARGA_BAJA_TCP) + f * np.array(C.GUARDADO_TCP)
            self.zr = _interp(C.CARGA_BAJA_Z, (0.90, 1.30), f)
            if f >= 1.0:
                self.estado, self.t = 'guardado', 0.0
        return ev

    def capsula_local(self):
        return Capsula(np.array([C.HOMBRO_X, 0.0]), self.tcp.copy(), C.GROSOR_BRAZO, self.zr)

    def capsula_mundo(self, pose):
        c = self.capsula_local()
        return Capsula(a_mundo(pose, c.p0), a_mundo(pose, c.p1), c.r, c.z, 'brazo')

    def tapa_lidar(self):
        return self.zr[0] <= C.LIDAR_Z <= self.zr[1] and self.tcp[0] > C.LIDAR_X + 0.05

    def sombra_mundo(self, pose):
        """Lo que tapa el lidar: el tramo final del brazo (muñeca + pinza, ~15 cm antes de la
        punta), que es lo que baja hasta el plano del lidar. El resto del brazo va más alto."""
        if not self.tapa_lidar():
            return None
        p1 = self.tcp
        u = p1 - np.array([C.HOMBRO_X, 0.0])
        u = u / max(float(np.hypot(*u)), 1e-6)
        p0 = p1 - 0.15 * u
        if p0[0] < C.LIDAR_X + 0.02:
            p0 = np.array([C.LIDAR_X + 0.02, p0[1]])
        return Capsula(a_mundo(pose, p0), a_mundo(pose, p1), C.GROSOR_BRAZO, (0.0, 2.0), 'brazo')

    def base_quieta(self):
        """Mientras el brazo alcanza/agarra o el lift sube, la base no se mueve."""
        return self.estado in ('alcanzando', 'agarrando', 'despejando')

    def punta_x(self):
        """Hasta dónde llega el robot hacia adelante con el brazo (frente de F(q))."""
        return max(C.FP_X_MAX, self.tcp[0] + C.GROSOR_BRAZO)

    def medio_ancho(self):
        return max(C.FP_HW, abs(self.tcp[1]) + C.GROSOR_BRAZO)

    def radio_giro(self):
        """r_giro(q): el punto de F(q) más lejos del eje de las ruedas."""
        r_base = max(math.hypot(x, C.FP_HW) for x in (C.FP_X_MIN, C.FP_X_MAX))
        return max(r_base, float(np.hypot(*self.tcp)) + C.GROSOR_BRAZO)

    def codigo(self):
        return ['guardado', 'alcanzando', 'agarrando', 'retrayendo', 'carga_baja',
                'despejando'].index(self.estado)


def _banda_alcance(z):
    """Alturas que ocupa el brazo cuando la pinza está en z (el hombro se pone lo más cerca)."""
    zh = min(max(z, C.HOMBRO_Z[0]), C.HOMBRO_Z[1])
    return (min(z, zh) - 0.06, max(z, zh) + 0.06)


def _interp(a, b, f):
    return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
