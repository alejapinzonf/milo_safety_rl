"""Adaptador a Gazebo: el MISMO filtro + escudo + observación del simulador 2D, alimentados con
sensores reales (scan, odometría) en vez de con el simulador.

Es Python puro (sin rclpy): el nodo ROS (ros/nodo_rl.py) solo le pasa mensajes y lee el comando.
Así se puede probar acá contra MiloEnv sin Gazebo (tests/test_adaptador.py).

Qué cambia respecto al entorno de entrenamiento (la validación sim-to-sim del paper):
  - la física, el lidar (500 rayos con ruido de Gazebo, que acá se remuestrea a los 180 del 2D) y
    la odometría son los de Gazebo;
  - el brazo va BLOQUEADO en la simulación de navegación: no hay punto ciego ni "despejar"
    (el punto ciego se valida solo en el 2D, ver E3b);
  - las personas (quietas) se detectan solo si el lidar las ve (un punto del scan a < 0.30 m de
    su posición), con el mismo ruido del detector que en el entrenamiento.
Las fases 'navegar' (Nav2) y 'orientar'/'acercar' (acá) son las del ejecutivo del 2D.
"""
import math
from types import SimpleNamespace

import numpy as np

from . import config as C
from .baselines import BASELINES
from .brazo import margen_alcance
from .entorno import MiloEnv
from .escudo import Theta, filtrar, proyectar
from .geometria import a_local, ang_norm
from .humanos import Detector

ALTO_OBJETO = {'botella': 0.22, 'taza': 0.095, 'galletas': 0.035}
DONDE_MUEBLE = {'laundry_table': 'mesa', 'dinner_table': 'redonda', 'shelf': 'estante',
                'bookshelf': 'estante'}


class Persona:
    """Una persona de la escena (posición real en el mundo; el robot solo ve lo que detecta)."""

    def __init__(self, ident, pos, vel=(0.0, 0.0)):
        self.id = ident
        self.pos = np.asarray(pos, float)
        self.vel = np.asarray(vel, float)

    def formas(self):  # no se usa (los choques los mide Gazebo), pero MiloEnv lo pide en viz
        return []


def objeto_desde_yaml(nombre, d):
    """Entrada de worlds/robocup_casa_objetos.yaml -> dict como el de la escena 2D."""
    donde = 'suelo' if d['donde'] == 'suelo' else DONDE_MUEBLE[d['mueble']]
    return dict(nombre=nombre, pos=np.array([d['x'], d['y']], float),
                z=float(d['z']) + ALTO_OBJETO[d['modelo']] / 2, donde=donde,
                forma=None, facil=False, prof_ef=d.get('prof_ef'))


class Adaptador(MiloEnv):
    def __init__(self, objeto, personas=(), cfg=None, semilla=0, modo='ppo'):
        super().__init__(cfg or C.ConfigEntorno(), semilla=semilla)
        self.modo = modo  # 'ppo' (la política decide) o el nombre de un baseline
        self.objeto = objeto
        self.personas = list(personas)
        self.esc = SimpleNamespace(objeto=objeto, humanos=self.personas, aprox={'W': np.zeros(2)},
                                   salida=np.zeros(2))
        self.reiniciar()

    # ------------------------------------------------------------------------------------------
    def reiniciar(self):
        sig = self.cfg.escenario
        self.obj_est = self.objeto['pos'] + self.rng.normal(0, sig.sigma_objeto, 2)
        self.det = Detector(sig.sigma_persona, sig.p_perder_persona, self.rng)
        self.brazo.reset()
        self.memoria.reset()
        self.pose = np.zeros(3)
        self.odom = self.pose.copy()
        self.s_odom = 0.0
        self.v = self.w = self.v_out = 0.0
        self.t = 0.0
        self.fase = 'navegar'
        self.accion_prev = np.zeros(5, np.float32)
        self.th = proyectar(Theta(), self.cfg.escudo)
        self.cmd = (0.0, 0.0)
        self.info_f = dict(d_frente=math.inf, d_atras=math.inf, intervino=False,
                           giro_bloqueado=False)
        self._frescos = set()
        self.alcanzado = False
        self.pts_obst = np.zeros((0, 2))
        self.rangos = np.full(C.LIDAR_RAYOS, C.LIDAR_RANGO)
        self.tapado = np.zeros(C.LIDAR_RAYOS, bool)
        self.d_mem = np.full(C.LIDAR_RAYOS, C.LIDAR_RANGO)
        self.m = self._metricas_cero()
        self.t_ini = 0.0

    def nueva_pierna(self, objeto, t_ini):
        """Empieza un episodio (ir a otro objeto) sin reiniciar el detector ni la pose."""
        self.objeto = objeto
        self.esc.objeto = objeto
        self.obj_est = objeto['pos'] + self.rng.normal(0, self.cfg.escenario.sigma_objeto, 2)
        self.fase = 'navegar'
        self.alcanzado = False
        self.t_ini = t_ini
        self.m = self._metricas_cero()

    def _obs(self):
        o = super()._obs()
        o[-1] = (self.t - getattr(self, 't_ini', 0.0)) / self.cfg.t_max  # tiempo de la pierna
        return o

    @staticmethod
    def _metricas_cero():
        return dict(exposicion=0.0, costo_soc=0.0, d_min_persona=math.inf, t_intimo=0.0,
                    t_personal=0.0, violaciones_iso=0, intrusiones_iso=0, intervenciones=0,
                    giros_bloqueados=0, recorrido=0.0, d_min_mueble=math.inf, choque=False)

    # ------------------------------------------------------------------------------------------
    # Sensores
    # ------------------------------------------------------------------------------------------
    def alimentar(self, t, pose, v, w, angulos, rangos, rango_min=0.05, rango_max=12.0):
        """pose = (x, y, yaw) de base_footprint en el mapa; v, w medidas (odometría);
        angulos/rangos = scan en el frame de base_footprint, respecto del lidar (0 = adelante)."""
        dt = max(t - self.t, 0.0) if self.t > 0 else 0.0
        self.t = t
        self.pose = np.array(pose, float)
        self.odom = self.pose.copy()
        self.v, self.w = float(v), float(w)
        self.m['recorrido'] += abs(self.v) * dt
        # remuestreo a los N rayos del 2D (el más cercano de cada sector angular) y limpieza
        N = C.LIDAR_RAYOS
        ang = np.asarray(angulos, float)
        r = np.asarray(rangos, float)
        ok = np.isfinite(r) & (r >= max(rango_min, 0.05)) & (r < min(rango_max, C.LIDAR_RANGO))
        k = np.round((ang[ok] + math.pi) / (2 * math.pi / N)).astype(int) % N
        d = np.full(N, C.LIDAR_RANGO)
        np.minimum.at(d, k, r[ok])
        dirs = self._dirs_loc
        o_loc = np.array([C.LIDAR_X, 0.0])
        pts = o_loc + dirs * d[:, None]
        hit = d < C.LIDAR_RANGO - 0.05
        # lo que cae sobre el propio robot (la columna) no es un obstáculo
        propio = ((pts[:, 0] >= C.FP_X_MIN - 0.02) & (pts[:, 0] <= C.FP_X_MAX + 0.02)
                  & (np.abs(pts[:, 1]) <= C.FP_HW + 0.02))
        hit &= ~propio
        d = np.where(propio, C.LIDAR_RANGO, d)
        self.rangos = d
        self.tapado = np.zeros(N, bool)
        self.d_mem = np.full(N, C.LIDAR_RANGO)
        self.pts_obst = pts[hit]
        # distancia mínima a muebles (para el reporte): de la huella a los puntos del lidar
        if len(self.pts_obst):
            x, y = self.pts_obst[:, 0], self.pts_obst[:, 1]
            dd = np.hypot(np.maximum(np.maximum(C.FP_X_MIN - x, 0), x - C.FP_X_MAX),
                          np.maximum(np.abs(y) - C.FP_HW, 0))
            self.m['d_min_mueble'] = min(self.m['d_min_mueble'], float(dd.min()))
            if dd.min() < 0.01:
                self.m['choque'] = True
        # personas: la ve el detector solo si el lidar tiene un punto sobre ella
        vistos = set()
        for h in self.personas:
            p = a_local(self.pose, h.pos)
            if len(self.pts_obst) and np.min(np.hypot(*(self.pts_obst - p).T)) < 0.30:
                vistos.add(h.id)
        self.det.actualizar(t, self.personas, vistos)

    # ------------------------------------------------------------------------------------------
    # Decisión
    # ------------------------------------------------------------------------------------------
    def politica(self, pol=None):
        """Elige θ (cada 0.2 s). pol(obs) -> acción en [-1, 1]^5, o un baseline por nombre."""
        if self.modo == 'ppo':
            a = np.clip(np.asarray(pol(self._obs()), np.float32), -1, 1)
        else:
            a = BASELINES[self.modo](self)
        th, _ = self.accion_a_theta(a)
        self.th = proyectar(th, self.cfg.escudo)
        self.accion_prev = a.astype(np.float32)
        return self.th

    def ejecutivo_final(self):
        """Fases 'orientar' y 'acercar' del ejecutivo 2D (sin brazo). -> (v, w) pedidos."""
        p = a_local(self.pose, self.obj_est)
        if self.fase == 'orientar':
            err = math.atan2(p[1], p[0])
            if abs(err) < 0.04:
                self.fase = 'acercar'
                return 0.0, 0.0
            return 0.0, float(np.clip(2.0 * err, -0.8, 0.8))
        if self.fase == 'acercar':
            if margen_alcance(p, self.objeto['z']) >= 0.0:
                self.fase = 'brazo'  # llegó: el brazo (bloqueado en Gazebo) agarraría desde acá
                self.alcanzado = True
                return 0.0, 0.0
            err = math.atan2(p[1], p[0])
            return 0.25, float(np.clip(1.5 * err, -0.5, 0.5))
        return 0.0, 0.0

    def margen(self):
        return margen_alcance(a_local(self.pose, self.objeto['pos']), self.objeto['z'])

    def filtrar_cmd(self, v_des, w_des, dt=0.05):
        """Pasa el comando pedido (Nav2 o ejecutivo final) por el filtro + escudo -> (v, w)."""
        self.cmd = (float(v_des), float(w_des))
        self._frescos = {i for i, tr in self.det.tracks.items() if self.t - tr['t'] <= 0.5}
        v, w, self.info_f = filtrar(v_des, w_des, self.th, self.pts_obst, self.brazo,
                                    self._personas_local(), self.cfg.escudo,
                                    self.det.hay_movimiento())
        self.v_out = v
        self.m['intervenciones'] += int(self.info_f['intervino'])
        self.m['giros_bloqueados'] += int(self.info_f['giro_bloqueado'])
        # costos de este paso (mismas fórmulas que en el 2D)
        rel = max(0.0, C.D_STOP_NOMINAL - (self.th.d_stop_f if self.v >= 0
                                           else self.th.d_stop_s)) / C.D_STOP_NOMINAL
        self.m['exposicion'] += abs(self.v) / C.V_MAX * rel * dt
        self.m['costo_soc'] += self._metricas_personas_dt(dt)
        return v, w

    def _metricas_personas_dt(self, dt):
        """_metricas_personas del 2D usa C.DT; acá el paso es el del timer del nodo."""
        a = C.DT
        C.DT = dt
        try:
            return self._metricas_personas()
        finally:
            C.DT = a


# ----------------------------------------------------------------------------------------------
class PoliticaNumpy:
    """El PPO sin torch: MLP 97 -> 128 -> 128 -> 5 con tanh (la media de la política)."""

    def __init__(self, ruta_npz):
        z = np.load(ruta_npz)
        self.W = [z['w0'], z['w1'], z['w2']]
        self.b = [z['b0'], z['b1'], z['b2']]

    def __call__(self, obs):
        h = np.asarray(obs, np.float32)
        for i in range(2):
            h = np.tanh(self.W[i] @ h + self.b[i])
        return np.clip(self.W[2] @ h + self.b[2], -1.0, 1.0)


def ang_dif(a, b):
    return ang_norm(a - b)
