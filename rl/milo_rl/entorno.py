"""Entorno gymnasium: un ciclo de agarre de Milo con el filtro de seguridad que ajusta el PPO.

Quién hace qué (igual que en el robot):
  - EJECUTIVO (este archivo, fijo): decide QUÉ hacer. Navega con A* + pure pursuit hasta frente al
    mueble, se acerca de frente al objeto hasta tenerlo al alcance, manda el brazo, retrocede y va
    a la salida. Pide velocidades (v, w).
  - FILTRO + ESCUDO (escudo.py): decide qué tan rápido puede ir sin chocar.
  - PPO (la política): cada 0.2 s elige los parámetros del filtro y si "despeja" (sube el brazo).
    No maneja el robot: solo decide qué tan cerca deja llegar y cuándo guardar el brazo.

Acción (5 números en [-1, 1]):
  0 d_stop adelante | 1 d_stop atrás/costados | 2 d_slow - d_stop | 3 v_max | 4 despejar (> 0)

Recompensa: agarrar +10, llegar a la salida +5, chocar -10, fallar -5, -0.01 por paso,
  shaping por acercarse a la sub-meta, castigo por moverse con el lidar tapado.
Costos (restricciones del CMDP, en info): 'costo_exp' (exposición: moverse con el filtro relajado)
  y 'costo_soc' (meterse a menos de 1.2 m de una persona). Con set_lambdas() se restan a la
  recompensa con sus multiplicadores (PPO-Lagrangiano, ver lagrangiano.py).
"""
import math

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    from . import _gym_minimo as gym

from . import config as C
from .brazo import Brazo, margen_alcance
from .escena import generar
from .escudo import Theta, dist_huella, filtrar, proyectar, separacion_iso
from .geometria import (Caja, Circulo, Trazador, a_local, a_mundo, ang_norm, chocan, distancia,
                        se_solapan_z)
from .humanos import Detector
from .memoria import Memoria

FASES = ['navegar', 'orientar', 'acercar', 'brazo', 'retroceder', 'salir']
N_SECT = 16
N_OBS = N_SECT * 3 + 6 + 4 + len(FASES) + 8 + 12 + 6 + 5 + 2
PROXEMIA = 1.2


class MiloEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, cfg=None, semilla=None):
        super().__init__()
        self.cfg = cfg or C.ConfigEntorno()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N_OBS,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.rng = np.random.default_rng(semilla)
        self.lambdas = (0.0, 0.0)
        ang = -math.pi + np.arange(C.LIDAR_RAYOS) * 2 * math.pi / C.LIDAR_RAYOS
        self._ang = ang
        self._dirs_loc = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        self._sector = ((ang + math.pi) / (2 * math.pi) * N_SECT).astype(int) % N_SECT
        self.brazo = Brazo()
        self.memoria = Memoria(self.cfg.escudo)

    def set_lambdas(self, l_exp, l_soc):
        self.lambdas = (float(l_exp), float(l_soc))

    def set_escenario(self, esc):
        self.cfg.escenario = esc

    def accion_a_theta(self, a):
        a = np.clip(np.asarray(a, float), -1, 1)
        R = self.cfg.accion

        def m(x, r):
            return r[0] + (x + 1) / 2 * (r[1] - r[0])
        th = Theta(m(a[0], R.d_stop), m(a[1], R.d_stop), m(a[2], R.d_slow_extra), m(a[3], R.v_max))
        if self.cfg.un_sector:
            th.d_stop_s = th.d_stop_f
        return th, bool(a[4] > 0)

    @staticmethod
    def theta_a_accion(th, despejar):
        R = C.RangoAccion()

        def inv(x, r):
            return 2 * (x - r[0]) / (r[1] - r[0]) - 1
        return np.array([inv(th.d_stop_f, R.d_stop), inv(th.d_stop_s, R.d_stop),
                         inv(th.d_slow_extra, R.d_slow_extra), inv(th.v_max, R.v_max),
                         1.0 if despejar else -1.0], np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        super().reset(seed=seed)
        self.esc = generar(self.rng, self.cfg.escenario)
        e = self.esc
        self.pose = np.array(e.inicio, float)
        self.odom = self.pose.copy()
        self.s_odom, self.v, self.w = 0.0, 0.0, 0.0
        self.t = 0.0
        self.brazo.reset()
        self.memoria.reset()
        self.det = Detector(self.cfg.escenario.sigma_persona, self.cfg.escenario.p_perder_persona,
                            self.rng)
        self.obj_est = e.objeto['pos'] + self.rng.normal(0, self.cfg.escenario.sigma_objeto, 2)
        self.estaticas = list(e.estaticas)
        self._prep_estaticas()
        self.ocultas_vistas = {}
        self._replan_oculta = False
        self.fase = 'navegar'
        self.camino = e.planificar(self.pose[:2], e.aprox['W'])
        self.k_camino = 0
        self.t_quieto = 0.0
        self.mejor_dist, self.t_sin_progreso = math.inf, 0.0
        self.x_agarre = None
        self.recup, self.recup_desde, self.t_bloq, self.t_replan = 0.0, None, 0.0, 0.0
        self.v_out = 0.0
        self.accion_prev = np.zeros(5, np.float32)
        self.th = proyectar(Theta(), self.cfg.escudo)
        self.cmd = (0.0, 0.0)
        self.info_f = dict(d_frente=math.inf, d_atras=math.inf, intervino=False,
                           giro_bloqueado=False)
        self.m = dict(exito=False, agarro=False, motivo='tiempo', t=0.0, choque_mueble=0,
                      contacto_persona=0, violaciones_iso=0, intrusiones_iso=0,
                      d_min_persona=math.inf,
                      t_intimo=0.0, t_personal=0.0, exposicion=0.0, costo_soc=0.0,
                      t_ciego_mov=0.0, intervenciones=0, recuperaciones=0, giros_bloqueados=0,
                      despejes=0,
                      tipo_personas=e.tipo_personas, donde=e.objeto['donde'],
                      facil=e.objeto['facil'], prof_ef=e.objeto['prof_ef'] or 0.0, recorrido=0.0,
                      oculta=any(f.tipo == 'caja_oculta' for f in e.estaticas), choque_oculta=0,
                      vio_oculta=0)
        self._scan()
        self.phi = self._potencial()
        return self._obs(), {}

    def _prep_estaticas(self):
        self.traz_est = Trazador(self.estaticas, C.LIDAR_Z)
        self._bbox = [(f, _centro(f), _radio(f)) for f in self.estaticas]

    def step(self, accion):
        accion = np.asarray(accion, np.float32)
        th, despejar = self.accion_a_theta(accion)
        self.th = proyectar(th, self.cfg.escudo)
        R = self.cfg.recompensa
        r, c_exp, c_soc = 0.0, 0.0, 0.0
        fin = False
        if despejar and self.brazo.despejar():
            self.m['despejes'] += 1
        for _ in range(C.SUBPASOS):
            v_des, w_des = self._ejecutivo()
            if self.brazo.base_quieta():
                v_des, w_des = 0.0, 0.0
            self.cmd = (v_des, w_des)
            self._frescos = {i for i, tr in self.det.tracks.items() if self.t - tr['t'] <= 0.5}
            v, w, self.info_f = filtrar(v_des, w_des, th, self.pts_obst, self.brazo,
                                        self._personas_local(), self.cfg.escudo,
                                        self.det.hay_movimiento())
            self._vigilar_bloqueo(v_des, w_des, v, w)
            self.v_out = v
            self.m['intervenciones'] += int(self.info_f['intervino'])
            self.m['giros_bloqueados'] += int(self.info_f['giro_bloqueado'])
            self._planta(v, w)
            ev = self.brazo.paso(C.DT)
            for h in self.esc.humanos:
                h.paso(C.DT, self._bloquea_humano)
            self.t += C.DT
            rel = max(0.0, C.D_STOP_NOMINAL - (self.th.d_stop_f if self.v >= 0
                                               else self.th.d_stop_s)) / C.D_STOP_NOMINAL
            c_exp += abs(self.v) / C.V_MAX * rel * C.DT
            frac = float(np.mean(self.tapado))
            r -= R.ciego * abs(self.v) / C.V_MAX * frac * C.DT * 10
            if frac > 0 and abs(self.v) > 0.02:
                self.m['t_ciego_mov'] += C.DT
            c_soc += self._metricas_personas()
            if self._choque():
                r += R.choque
                self.m['motivo'] = 'choque'
                fin = True
                break
            if ev == 'agarro':
                if margen_alcance(a_local(self.pose, self.esc.objeto['pos']),
                                  self.esc.objeto['z']) >= -0.02:
                    r += R.agarrar
                    self.m['agarro'] = True
                    self._iniciar('retroceder')
                    if self.esc.objeto['forma'] is not None:
                        self.estaticas = [f for f in self.estaticas
                                          if f is not self.esc.objeto['forma']]
                        self._prep_estaticas()
                else:
                    r += R.fallar
                    self.m['motivo'] = 'agarre_fallido'
                    fin = True
                    break
            falla = self._revisar_progreso()
            if falla:
                r += R.fallar
                self.m['motivo'] = falla
                fin = True
                break
            if self.fase == 'salir' and np.hypot(*(self.pose[:2] - self.esc.salida)) < 0.3:
                r += R.llegar
                self.m['exito'] = True
                self.m['motivo'] = 'exito'
                fin = True
                break
            self._scan()
        phi = self._potencial()
        r += R.progreso * (phi - self.phi) + R.paso
        self.phi = phi
        self.m['exposicion'] += c_exp
        self.m['costo_soc'] += c_soc
        self.m['t'] = self.t
        r_pen = r - self.lambdas[0] * c_exp - self.lambdas[1] * c_soc
        trunc = (not fin) and self.t >= self.cfg.t_max
        info = dict(costo_exp=c_exp, costo_soc=c_soc, r_tarea=r)
        if fin or trunc:
            info['episodio'] = dict(self.m)
        self.accion_prev = accion
        return self._obs(), float(r_pen), fin, trunc, info

    def _planta(self, v, w):
        def lim(x, x_des, a_sube):
            freno = abs(x_des) < abs(x) or x_des * x < 0
            a = C.A_FRENO if freno else a_sube
            return x + float(np.clip(x_des - x, -a * C.DT, a * C.DT))
        self.v = lim(self.v, v, C.A_LIN)
        self.w = lim(self.w, w, C.A_ANG)
        x, y, psi = self.pose
        self.pose = np.array([x + self.v * math.cos(psi) * C.DT,
                              y + self.v * math.sin(psi) * C.DT, ang_norm(psi + self.w * C.DT)])
        self.m['recorrido'] += abs(self.v) * C.DT
        vo = self.v * (1 + self.rng.normal(0, 0.01))
        wo = self.w * (1 + self.rng.normal(0, 0.02)) + self.rng.normal(0, 0.002)
        xo, yo, po = self.odom
        self.odom = np.array([xo + vo * math.cos(po) * C.DT, yo + vo * math.sin(po) * C.DT,
                              ang_norm(po + wo * C.DT)])
        self.s_odom += abs(vo) * C.DT

    def _formas_robot(self):
        cx = (C.FP_X_MIN + C.FP_X_MAX) / 2
        base = Caja(a_mundo(self.pose, [cx, 0.0]), ((C.FP_X_MAX - C.FP_X_MIN) / 2, C.FP_HW),
                    self.pose[2], (0.0, C.BASE_Z), 'robot')
        return base, self.brazo.capsula_mundo(self.pose)

    def _bloquea_humano(self, p):
        base, cap = self._formas_robot()
        q = Circulo(np.asarray(p), 0.2, (0, 2))
        return distancia(q, base) < 0.15 or distancia(q, cap) < 0.15

    def _choque(self):
        base, cap = self._formas_robot()
        pc = self.pose[:2]
        mover_brazo = self.brazo.estado not in ('alcanzando', 'agarrando', 'retrayendo')
        for f, c, rad in self._bbox:
            if np.hypot(*(c - pc)) - rad > 1.6:
                continue
            if chocan(base, f):
                self.m['choque_mueble'] += 1
                self.m['choque_oculta'] += int(f.tipo == 'caja_oculta')
                return True
            if mover_brazo and f.tipo not in ('mueble_obj', 'objeto') and chocan(cap, f):
                self.m['choque_mueble'] += 1
                self.m['choque_oculta'] += int(f.tipo == 'caja_oculta')
                return True
        for h in self.esc.humanos:
            if np.hypot(*(h.pos - pc)) > 1.6:
                continue
            for f in h.formas():
                if chocan(base, f) or chocan(cap, f):
                    self.m['contacto_persona'] += 1
                    return True
        return False

    def _scan(self):
        """Lidar: rayos contra muebles + personas + la sombra del brazo."""
        o = a_mundo(self.pose, [C.LIDAR_X, 0.0])
        dirs = self._dirs_loc @ np.array([[math.cos(self.pose[2]), math.sin(self.pose[2])],
                                          [-math.sin(self.pose[2]), math.cos(self.pose[2])]])
        d, ids = self.traz_est.tirar(o, dirs, C.LIDAR_RANGO)
        din = [f for h in self.esc.humanos for f in h.formas()]
        sombra = self.brazo.sombra_mundo(self.pose)
        if sombra is not None:
            din.append(sombra)
        tapado = np.zeros(len(d), bool)
        vistos = set()
        if din:
            tr = Trazador(din, C.LIDAR_Z)
            d2, id2 = tr.tirar(o, dirs, C.LIDAR_RANGO)
            mas_cerca = d2 < d
            for k in np.nonzero(mas_cerca)[0]:
                f = tr.formas[id2[k]]
                if f.tipo == 'brazo':
                    tapado[k] = True
                elif f.tipo == 'persona':
                    vistos.add(f.ident)
            d = np.where(mas_cerca, d2, d)
        d = d + self.rng.normal(0, C.LIDAR_SIGMA, len(d))
        self.tapado = tapado
        for k in np.nonzero((ids >= 0) & ~tapado & (d < C.LIDAR_RANGO - 0.05))[0]:
            f = self.traz_est.formas[ids[k]]
            if f.tipo == 'caja_oculta' and id(f) not in self.ocultas_vistas:
                self.ocultas_vistas[id(f)] = f.c
                self._replan_oculta = self.cfg.escenario.planificador_ve_oculta
                self.m['vio_oculta'] = 1
        ok = ~tapado
        o_loc = np.array([C.LIDAR_X, 0.0])
        pts_loc = o_loc + self._dirs_loc * d[:, None]
        hit = ok & (d < C.LIDAR_RANGO - 0.05)
        pts_odom = a_mundo(self.odom, pts_loc[ok])
        self.memoria.guardar(pts_odom, self.t, self.s_odom, self.odom[2])
        self.det.actualizar(self.t, self.esc.humanos, vistos)
        obst = [pts_loc[hit]]
        self.d_mem = np.full(C.LIDAR_RAYOS, C.LIDAR_RANGO)
        if tapado.any() and self.cfg.escudo.activo and self.cfg.escudo.usar_memoria:
            idx = np.nonzero(tapado)[0]
            vp, dk, _ = self.memoria.virtuales(self.odom, idx, self.t, self.s_odom, self.odom[2],
                                               self.det.hay_movimiento(), self.brazo.punta_x())
            obst.append(vp)
            self.d_mem[idx] = dk
        self.pts_obst = np.concatenate(obst) if obst else np.zeros((0, 2))
        self.rangos = np.where(ok, d, 0.0)

    def _personas_local(self):
        out = []
        for pos, vel, edad, cota in self.det.estimaciones(self.t):
            out.append((a_local(self.pose, pos), vel @ _R(self.pose[2]), edad, cota))
        return out

    def _metricas_personas(self):
        """Métricas con las posiciones REALES de las personas. Devuelve el costo social."""
        if not self.esc.humanos:
            return 0.0
        costo = 0.0
        punta, hw = self.brazo.punta_x(), self.brazo.medio_ancho()
        for h in self.esc.humanos:
            p = a_local(self.pose, h.pos)
            d = dist_huella(p, C.FP_X_MIN, punta, hw)
            self.m['d_min_persona'] = min(self.m['d_min_persona'], d)
            if d < PROXEMIA:
                costo += (PROXEMIA - d) / PROXEMIA * (0.2 + abs(self.v) / C.V_MAX) * C.DT
                self.m['t_personal'] += C.DT
            if d < 0.45:
                self.m['t_intimo'] += C.DT
            hacia = (self.v_out > 0.02 and p[0] > 0) or (self.v_out < -0.02 and p[0] < 0)
            if hacia:
                s_p = separacion_iso(abs(self.v_out), float(np.hypot(*h.vel)), self.cfg.escudo,
                                     con_detector=False)
                if d < s_p:
                    self.m['intrusiones_iso'] += 1
                    if h.id in self._frescos:
                        self.m['violaciones_iso'] += 1
        return costo

    def _iniciar(self, fase):
        self.fase = fase
        self.t_quieto = 0.0
        self.mejor_dist, self.t_sin_progreso = math.inf, 0.0
        if fase == 'retroceder':
            self.x_agarre = self.pose[:2].copy()
        if fase == 'salir':
            self._replanificar()
        self.phi = self._potencial()

    def _ejecutivo(self):
        f = self.fase
        if f in ('navegar', 'salir') and self.recup > 0:
            return self._recuperar()
        if f in ('navegar', 'salir') and self._replan_oculta:
            self._replanificar()
            self._replan_oculta = False
        if f in ('navegar', 'salir') and self.esc.humanos and not self.det.hay_movimiento():
            self.t_replan += C.DT
            if self.t_replan > 2.0:
                self._replanificar()
        if f == 'navegar':
            if np.hypot(*(self.pose[:2] - self.esc.aprox['W'])) < 0.15:
                self._iniciar('orientar')
                return 0.0, 0.0
            return self._seguir(0.35)
        if f == 'orientar':
            p = a_local(self.pose, self.obj_est)
            err = math.atan2(p[1], p[0])
            if abs(err) < 0.04:
                self._iniciar('acercar')
                return 0.0, 0.0
            return 0.0, float(np.clip(2.0 * err, -0.8, 0.8))
        if f == 'acercar':
            p = a_local(self.pose, self.obj_est)
            if margen_alcance(p, self.esc.objeto['z']) >= 0.0:
                self._iniciar('brazo')
                self.brazo.alcanzar(p, self.esc.objeto['z'])
                return 0.0, 0.0
            err = math.atan2(p[1], p[0])
            return 0.25, float(np.clip(1.5 * err, -0.5, 0.5))
        if f == 'brazo':
            return 0.0, 0.0
        if f == 'retroceder':
            if np.hypot(*(self.pose[:2] - self.x_agarre)) > 0.6:
                self._iniciar('salir')
                return 0.0, 0.0
            return -0.2, 0.0
        if f == 'salir':
            if self.camino is None:
                self.camino = self.esc.planificar(self.pose[:2], self.esc.salida)
                if self.camino is None:
                    return 0.0, 0.0
            return self._seguir(0.35)
        return 0.0, 0.0

    def _recuperar(self):
        """Recuperación (como el BackUp de Nav2): retrocede 25 cm y vuelve a planificar."""
        self.recup -= C.DT
        if self.recup <= 0 or np.hypot(*(self.pose[:2] - self.recup_desde)) > 0.25:
            self.recup = 0.0
            self._replanificar()
            return 0.0, 0.0
        return -0.15, 0.0

    def _replanificar(self):
        meta = self.esc.aprox['W'] if self.fase == 'navegar' else self.esc.salida
        evitar = [] if self.det.hay_movimiento() else \
            [(q[0], 0.9) for q in self.det.estimaciones(self.t) if q[2] < 1.0]
        if self.cfg.escenario.planificador_ve_oculta:
            evitar += [(c, 0.65) for c in self.ocultas_vistas.values()]
        self.camino = self.esc.planificar(self.pose[:2], meta, evitar)
        self.k_camino = 0
        self.t_replan = 0.0

    def _vigilar_bloqueo(self, v_des, w_des, v, w):
        """Si el filtro no lo deja hacer lo que pide el ejecutivo por 1.5 s, recuperación."""
        if self.fase not in ('navegar', 'salir') or self.recup > 0:
            self.t_bloq = 0.0
            return
        pide = abs(v_des) > 0.02 or abs(w_des) > 0.05
        hace = abs(v) > 0.01 or abs(w) > 0.02
        persona = any(np.hypot(*q[0]) < 2.0 for q in self._personas_local() if q[2] < 0.5)
        self.t_bloq = self.t_bloq + C.DT if (pide and not hace and not persona) else 0.0
        if self.t_bloq > 1.5:
            self.t_bloq = 0.0
            self.recup, self.recup_desde = 2.5, self.pose[:2].copy()
            self.m['recuperaciones'] += 1

    def _seguir(self, v_nom):
        """Pure pursuit sobre el camino de A*."""
        if not self.camino:
            return 0.0, 0.0
        pos = self.pose[:2]
        while self.k_camino < len(self.camino) - 1 and \
                np.hypot(*(self.camino[self.k_camino] - pos)) < 0.45:
            self.k_camino += 1
        meta = self.camino[self.k_camino]
        p = a_local(self.pose, meta)
        alfa = math.atan2(p[1], p[0])
        if abs(alfa) > 0.7:
            return 0.0, math.copysign(0.8, alfa)
        dist = np.hypot(*(self.camino[-1] - pos))
        v = v_nom * (1 - abs(alfa) / 1.2) * min(1.0, dist / 0.5 + 0.2)
        return v, float(np.clip(2.0 * alfa, -W_EJEC, W_EJEC))

    def _revisar_progreso(self):
        """Falla si se queda atascado (el filtro no lo deja avanzar o algo le tapa el paso)."""
        f = self.fase
        if f in ('brazo', 'orientar'):
            return None
        if f == 'acercar':
            if abs(self.v) < 0.01:
                self.t_quieto += C.DT
                if self.t_quieto > 3.0:
                    return 'no_alcanza'
            else:
                self.t_quieto = 0.0
            return None
        meta = {'navegar': self.esc.aprox['W'], 'salir': self.esc.salida,
                'retroceder': None}[f]
        if meta is None:
            if abs(self.v) < 0.01:
                self.t_quieto += C.DT
                if self.t_quieto > 3.0:
                    self._iniciar('salir')
            return None
        d = float(np.hypot(*(self.pose[:2] - meta)))
        if d < self.mejor_dist - 0.05:
            self.mejor_dist, self.t_sin_progreso = d, 0.0
        else:
            self.t_sin_progreso += C.DT
            if self.t_sin_progreso > 15.0:
                return 'atascado'
        return None

    def _potencial(self):
        f = self.fase
        if f == 'navegar':
            return -float(np.hypot(*(self.pose[:2] - self.esc.aprox['W'])))
        if f in ('orientar', 'acercar'):
            p = a_local(self.pose, self.obj_est)
            return min(margen_alcance(p, self.esc.objeto['z']), 0.0)
        if f == 'salir':
            return -float(np.hypot(*(self.pose[:2] - self.esc.salida)))
        return 0.0

    def _obs(self):
        o = []
        rng_max = 4.0
        vis = np.where(self.tapado, rng_max, np.minimum(self.rangos, rng_max))
        sect = np.full(N_SECT, rng_max)
        np.minimum.at(sect, self._sector, vis)
        tap = np.bincount(self._sector, self.tapado.astype(float), N_SECT) / \
            np.bincount(self._sector, minlength=N_SECT)
        mem = np.full(N_SECT, rng_max)
        np.minimum.at(mem, self._sector, np.minimum(self.d_mem, rng_max))
        o += list(sect / rng_max) + list(tap) + list(mem / rng_max)
        b = self.brazo
        oh = np.zeros(6)
        oh[b.codigo()] = 1
        o += list(oh)
        o += [b.tcp[0] / 0.7, b.tcp[1] / 0.5, (b.punta_x() - C.FP_X_MAX) / 0.6, b.radio_giro()]
        fz = np.zeros(len(FASES))
        fz[FASES.index(self.fase)] = 1
        o += list(fz)
        p = a_local(self.pose, self.obj_est)
        ob = self.esc.objeto
        donde = np.zeros(4)
        donde[['mesa', 'redonda', 'estante', 'suelo'].index(ob['donde'])] = 1
        o += [p[0] / 3, p[1] / 3, ob['z'] / 2, float(np.clip(margen_alcance(p, ob['z']), -2, 1))]
        o += list(donde)
        per = np.zeros(12)
        if self.cfg.usar_personas_en_obs:
            est = sorted(self._personas_local(), key=lambda q: np.hypot(*q[0]))[:2]
            for i, (pos, vel, edad, _) in enumerate(est):
                per[i * 6:(i + 1) * 6] = [np.clip(pos[0] / 5, -2, 2), np.clip(pos[1] / 5, -2, 2),
                                          vel[0] / 1.5, vel[1] / 1.5, min(edad, 5) / 5, 1.0]
        o += list(per)
        df = min(self.info_f['d_frente'], rng_max) / rng_max
        da = min(self.info_f['d_atras'], rng_max) / rng_max
        o += [self.cmd[0] / C.V_MAX, self.cmd[1] / C.W_MAX, self.v / C.V_MAX, self.w / C.W_MAX,
              df, da]
        o += list(self.accion_prev)
        o += [float(self.det.hay_movimiento()), self.t / self.cfg.t_max]
        return np.asarray(o, np.float32)


W_EJEC = 0.8


def _R(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _centro(f):
    return np.asarray(f.c if hasattr(f, 'c') else (f.p0 + f.p1) / 2, float)


def _radio(f):
    if hasattr(f, 'half'):
        return float(np.hypot(*f.half))
    if hasattr(f, 'r') and hasattr(f, 'c'):
        return f.r
    return float(np.hypot(*(f.p1 - f.p0))) / 2 + f.r


__all__ = ['MiloEnv', 'N_OBS', 'FASES', 'se_solapan_z']
