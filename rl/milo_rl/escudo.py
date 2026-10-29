"""Filtro de seguridad parametrizado (lo que ajusta el PPO) + escudo certificado (lo que no).

Es el mismo filtro de andesrobot_safety/logic.py (corredor adelante/atrás, giro en el lugar) con
cuatro cambios, que son el aporte del paper:

1. RAMPA SIGMOIDE. La velocidad baja de forma suave entre d_stop y d_slow:
       s(d) = 0                                   si d <= d_stop
       s(d) = (sigma(k (d - d_mid)) - 0.05) / 0.9  (recortado a [0, 1])
       d_mid = (d_stop + d_slow) / 2,  k = 2 ln(19) / (d_slow - d_stop)
   (sigma vale 0.05 en d_stop y 0.95 en d_slow, así que s va de 0 a 1 justo en ese tramo).
   d_stop adelante y d_stop atrás/costados son distintos: dos sectores.

2. VELOCIDAD MÁXIMA SEGURA frente a muebles (Proposición 1). Frenando con a y reaccionando en T:
       v <= v_cap(d) = -a T + sqrt(a² T² + 2 a (d - d_min))
   Si en cada paso v <= v_cap(d), el borde del robot nunca queda a menos de d_min de un mueble.
   Además el escudo proyecta los parámetros: d_stop >= d_min, d_slow >= d_stop + 0.05.

3. SEPARACIÓN ISO/TS 15066 frente a personas (Proposición 2):
       S_p(v_r) = v_h (T_r + T_s) + v_r T_r + v_r² / (2a) + C + Z_d + Z_r
   Se exige d_h >= S_p(v_r), de donde sale otra v_cap. El PPO no puede tocar este piso.
   v_h = velocidad estimada de la persona (+ V_HUMANO_MAX * edad si se dejó de ver y hay gente
   moviéndose).

4. HUELLA Y GIRO QUE DEPENDEN DEL BRAZO (Proposición 4). El frente del corredor es la punta de
   F(q) (con el brazo afuera, más adelante), el ancho es el de F(q), y girar solo se permite si
   no hay nada dentro de r_giro(q) + holgura. Los rayos tapados por el brazo se reemplazan por los
   puntos de la memoria (Proposición 3, ver memoria.py).

Con params.activo = False queda el filtro de fábrica (ablación "sin escudo"): sin v_cap, sin ISO,
sin memoria (lo tapado se ignora) y con el radio de giro de la base sola.
"""
import math
from dataclasses import dataclass

import numpy as np

from . import config as C

LN19 = math.log(19.0)
EDAD_MAX = 0.5


@dataclass
class Theta:
    """Parámetros del filtro que elige el PPO (ya en unidades físicas)."""

    d_stop_f: float = C.D_STOP_NOMINAL
    d_stop_s: float = C.D_STOP_NOMINAL
    d_slow_extra: float = C.D_SLOW_NOMINAL - C.D_STOP_NOMINAL
    v_max: float = C.V_MAX


def rampa(d, d_stop, d_slow):
    if d <= d_stop:
        return 0.0
    if d >= d_slow:
        return 1.0
    k = 2 * LN19 / max(d_slow - d_stop, 1e-6)
    sg = 1.0 / (1.0 + math.exp(-k * (d - 0.5 * (d_stop + d_slow))))
    return min(max((sg - 0.05) / 0.9, 0.0), 1.0)


def v_cap(d, d_min, a, T):
    """Velocidad con la que todavía se alcanza a frenar antes de quedar a d_min."""
    return -a * T + math.sqrt(a * a * T * T + 2 * a * max(d - d_min, 0.0))


def proyectar(th, p):
    """Lleva theta al conjunto seguro (fórmula cerrada: recortes)."""
    if not p.activo:
        return th
    ds_f = max(th.d_stop_f, p.d_min_mueble)
    ds_s = max(th.d_stop_s, p.d_min_mueble)
    return Theta(ds_f, ds_s, max(th.d_slow_extra, 0.05), min(th.v_max, C.V_MAX))


def corredor(pts, x_min, punta, hw):
    """Distancia libre adelante (desde la punta de F(q)) y atrás (desde la cola) en el corredor."""
    if len(pts) == 0:
        return math.inf, math.inf
    x, y = pts[:, 0], pts[:, 1]
    en = np.abs(y) <= hw
    fr = x[en & (x > punta)] - punta
    at = x_min - x[en & (x < x_min)]
    return (float(fr.min()) if fr.size else math.inf, float(at.min()) if at.size else math.inf)


def dist_huella(p, x_min, punta, hw):
    """Distancia de un punto (frame robot) a la caja [x_min, punta] x [-hw, hw]."""
    dx = max(x_min - p[0], 0.0, p[0] - punta)
    dy = max(abs(p[1]) - hw, 0.0)
    return math.hypot(dx, dy)


def _v_h(vel, edad, cota, hay_movimiento):
    """Velocidad de la persona que usa el escudo.
    - Si en la escena hay gente caminando, se supone la máxima, V_HUMANO_MAX, para todos (como la
      ISO, que usa una velocidad humana fija: la estimada llega tarde y subestima), más lo que pudo
      avanzar si hace rato que no se ve.
    - Si nadie camina: la estimada; pero a alguien recién detectado (sin velocidad confiable
      todavía, cota = V_HUMANO_MAX) se le supone la máxima hasta tener datos."""
    if hay_movimiento:
        return C.V_HUMANO_MAX * (1.0 + edad)
    v = float(np.hypot(*vel))
    return cota if cota >= C.V_HUMANO_MAX else v


def separacion_iso(v_r, v_h, p, con_detector=True):
    """S_p de la ISO/TS 15066. con_detector=False: sin Z_d, para medir con la posición REAL de la
    persona (Z_d está para cubrir el error del detector, que con la verdad no existe)."""
    return (v_h * (p.t_reaccion + p.t_parada) + v_r * p.t_reaccion
            + v_r * v_r / (2 * p.a_frenado) + p.c_intrusion + p.z_robot
            + (p.z_detector if con_detector else 0.0))


def filtrar(v, w, th, pts, brazo, personas, p, hay_movimiento=False):
    """Devuelve (v, w, info) seguros.

    pts: obstáculos (frame robot): lidar no tapado + puntos virtuales de la memoria.
    personas: lista de (pos (2,), vel (2,), edad [s], v_cota) del detector (frame robot).
      Solo cuentan las vistas hace poco (EDAD_MAX). Una persona que no se ve y no está en la sombra
      del brazo está detrás de un mueble: si entrara al corredor, el lidar la vería. Las que
      pudieran estar en la sombra del brazo las cubre la memoria (erosión).
    """
    personas = [q for q in personas if q[2] <= EDAD_MAX]
    th = proyectar(th, p)
    info = dict(d_frente=math.inf, d_atras=math.inf, intervino=False, giro_bloqueado=False,
                v_rampa=0.0, theta=th)
    v = max(-th.v_max, min(th.v_max, v))
    w = max(-C.W_MAX, min(C.W_MAX, w))
    if p.activo:
        punta, hw, r_giro = brazo.punta_x(), brazo.medio_ancho(), brazo.radio_giro()
    else:
        punta, hw = C.FP_X_MAX, C.FP_HW
        r_giro = max(math.hypot(x, C.FP_HW) for x in (C.FP_X_MIN, C.FP_X_MAX))
    hw_c = hw + p.margen_lateral
    d_f, d_a = corredor(pts, C.FP_X_MIN, punta, hw_c)
    info['d_frente'], info['d_atras'] = d_f, d_a
    if v > 0:
        v *= rampa(d_f, th.d_stop_f, th.d_stop_f + th.d_slow_extra)
    elif v < 0:
        v *= rampa(d_a, th.d_stop_s, th.d_stop_s + th.d_slow_extra)
    info['v_rampa'] = v
    if p.activo:
        d = d_f if v > 0 else d_a
        cap = v_cap(d, p.d_min_mueble, p.a_frenado, p.t_reaccion)
        for pos, vel, edad, cota in personas:
            if (v > 0 and pos[0] <= -0.3) or (v < 0 and pos[0] >= 0.3) or v == 0:
                continue
            v_h = _v_h(vel, edad, cota, hay_movimiento)
            d_h = dist_huella(pos, C.FP_X_MIN, punta, hw)
            resto = d_h - v_h * (p.t_reaccion + p.t_parada + C.DT) - p.c_intrusion \
                - p.z_detector - p.z_robot
            cap = min(cap, v_cap(resto, 0.0, p.a_frenado, p.t_reaccion + C.DT))
        if abs(v) > cap:
            v = math.copysign(cap, v)
            info['intervino'] = True
    brazo_afuera = p.activo and punta > C.FP_X_MAX + 1e-6
    if w != 0.0 and (abs(v) < 0.02 or brazo_afuera):
        r_min = float(np.hypot(pts[:, 0], pts[:, 1]).min()) if len(pts) else math.inf
        bloquea = r_min < r_giro + p.holgura_giro
        if p.activo:
            for pos, vel, edad, cota in personas:
                v_h = _v_h(vel, edad, cota, hay_movimiento)
                if float(np.hypot(*pos)) - r_giro < separacion_iso(0.0, v_h, p):
                    bloquea = True
        if bloquea:
            w = 0.0
            info['giro_bloqueado'] = True
    return v, w, info
