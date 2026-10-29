"""Lógica pura del filtro de seguridad (sin ROS, testeable).

Todo en el frame base_footprint: +X adelante, +Y izquierda, origen = centro del eje de ruedas.

Por qué está separado de safety_filter.py: aquí solo hay matemática (recibe números, devuelve
números). Así se puede probar con pytest sin tener ROS corriendo (ver test/test_logic.py).

Vista desde arriba (x hacia la derecha = adelante de Milo):

              corredor (ancho = robot + side_margin a cada lado)
     ┌──────────────┬─────────────────────┬──────────────┐
     │ zona trasera │       ROBOT         │ zona delant. │
     └──────────────┴─────────────────────┴──────────────┘
                 x_min (-0.12)     (0,0)  x_max (0.44)
  Solo cuentan los obstáculos DENTRO del corredor y en la dirección en que se mueve.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyParams:
    footprint_x_min: float = -0.12
    footprint_x_max: float = 0.44
    footprint_half_width: float = 0.25
    stop_distance: float = 0.3
    slow_distance: float = 0.7
    side_margin: float = 0.10
    rotation_clearance: float = 0.15
    max_linear: float = 0.4
    max_angular: float = 1.0
    turn_check_only_in_place: bool = False


def robot_radius(p: SafetyParams) -> float:
    """Radio del círculo que barre el robot al girar en el lugar."""
    xs = (p.footprint_x_min, p.footprint_x_max)
    return max(math.hypot(x, p.footprint_half_width) for x in xs)


def corridor_distances(pts: np.ndarray, p: SafetyParams):
    """Distancia libre adelante y atrás (desde el borde del robot) dentro del corredor."""
    if pts.size == 0:
        return math.inf, math.inf
    x, y = pts[:, 0], pts[:, 1]
    in_lane = np.abs(y) <= p.footprint_half_width + p.side_margin
    front = x[in_lane & (x > p.footprint_x_max)] - p.footprint_x_max
    rear = p.footprint_x_min - x[in_lane & (x < p.footprint_x_min)]
    return (float(front.min()) if front.size else math.inf,
            float(rear.min()) if rear.size else math.inf)


def remove_self(pts: np.ndarray, p: SafetyParams) -> np.ndarray:
    """Descarta puntos que caen sobre el propio robot (p. ej. la columna)."""
    if pts.size == 0:
        return pts
    inside = ((pts[:, 0] >= p.footprint_x_min - 0.02) & (pts[:, 0] <= p.footprint_x_max + 0.02)
              & (np.abs(pts[:, 1]) <= p.footprint_half_width + 0.02))
    return pts[~inside]


def scale_for(d: float, p: SafetyParams) -> float:
    """1 = libre, 0 = parar; lineal entre stop_distance y slow_distance."""
    if d <= p.stop_distance:
        return 0.0
    if d >= p.slow_distance:
        return 1.0
    return (d - p.stop_distance) / (p.slow_distance - p.stop_distance)


def filter_cmd(v: float, w: float, pts: np.ndarray, p: SafetyParams):
    """Devuelve (v, w, estado) seguros. pts: Nx2 en base_footprint, sin el robot."""
    v = max(-p.max_linear, min(p.max_linear, v))
    w = max(-p.max_angular, min(p.max_angular, w))
    d_front, d_rear = corridor_distances(pts, p)
    state = 'libre'

    if v > 0.0:
        s = scale_for(d_front, p)
        if s < 1.0:
            state = 'PARADO adelante (%.2f m)' % d_front if s == 0.0 else 'frenando adelante (%.2f m)' % d_front
        v *= s
    elif v < 0.0:
        s = scale_for(d_rear, p)
        if s < 1.0:
            state = 'PARADO atrás (%.2f m)' % d_rear if s == 0.0 else 'frenando atrás (%.2f m)' % d_rear
        v *= s

    turning_in_place = abs(v) < 0.02
    check_turn = turning_in_place or not p.turn_check_only_in_place
    if w != 0.0 and pts.size and check_turn:
        r_min = float(np.hypot(pts[:, 0], pts[:, 1]).min())
        if r_min < robot_radius(p) + p.rotation_clearance:
            w = 0.0
            state += ' | giro bloqueado (%.2f m)' % r_min
    return v, w, state
