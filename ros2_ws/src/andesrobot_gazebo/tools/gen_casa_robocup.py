#!/usr/bin/env python3
"""Genera la casa RoboCup@Home (Laundry / Bedroom / Living room / Kitchen) para Gazebo Classic.

UNA sola fuente de verdad: aquí están los muros, las puertas y los muebles. Desde esto se generan
2 archivos (no los edites a mano, edita este script y córrelo):

  andesrobot_gazebo/worlds/robocup_casa.world   el mundo SDF para Gazebo
  andesrobot_gazebo/worlds/robocup_casa.png     vista de arriba con los números del rulebook

Además revisa que Milo quepa por todas las puertas y pueda acercarse a cada mueble del rulebook
(sin chocar, sin que el filtro de seguridad lo frene y pudiendo girar en el lugar).

Uso (en tu PC o dentro del contenedor, solo necesita python3 + numpy + pillow):
  python3 ros2_ws/src/andesrobot_gazebo/tools/gen_casa_robocup.py
  ./sim.sh build          # para que ROS vea el mundo nuevo

Medidas: sacadas del plano del rulebook (imagen "Locations"), escala ajustada para que la casa
mida 9 x 9 m y la cama ~2 m. Puertas de 0.90 m (como una puerta real de casa).

SISTEMAS DE COORDENADAS
  "casa" (u, v): origen en la esquina exterior abajo-izquierda del plano,
                u -> derecha, v -> arriba.
  "mundo" (x, y): el de Gazebo, con el (0, 0) en el CENTRO de la casa. Milo parte afuera de la
  puerta del living mirando hacia adentro (+x); esa pose la pasa sim.sh (CASA_SPAWN). En la sim
  la odometría sale de la pose real en Gazebo, así que el frame "map" del SLAM queda igual al
  mundo: las poses que anotas con tf2_echo son coordenadas de Gazebo.
"""

import math
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_GAZEBO = os.path.dirname(HERE)
SRC = os.path.dirname(PKG_GAZEBO)
OUT_WORLD = os.path.join(PKG_GAZEBO, 'worlds', 'robocup_casa.world')
OUT_PNG = os.path.join(PKG_GAZEBO, 'worlds', 'robocup_casa.png')
OUT_OBJ = os.path.join(PKG_GAZEBO, 'worlds', 'robocup_casa_objetos.yaml')
LUGARES = os.path.join(SRC, 'andesrobot_slam', 'maps', 'casa_lugares.yaml')

ESCALA = 1.4
L = 9.0
T = 0.15
WALL_H = 1.2
DOOR = 1.40
SPAWN_UV = (-0.9, 3.17)
LIDAR_Z = 0.363
FALDONES = True
FALDON_Z = (0.15, 0.45)
APPROACH = 0.80
FP = [(0.44, 0.25), (0.44, -0.25), (-0.12, -0.25), (-0.12, 0.25)]
TURN_R = math.hypot(0.44, 0.25) + 0.08

XV = 4.475
YH = 5.155


def to_world(u, v):
    return (round(u - L / 2, 3), round(v - L / 2, 3))


WALL_LINES = [
    ('h', T / 2, 0.0, L, []),
    ('h', L - T / 2, 0.0, L, []),
    ('v', L - T / 2, T, L - T, []),
    ('v', T / 2, T, L - T, [3.17, 7.50]),
    ('v', XV, T, L - T, [2.73, 8.06]),
    ('h', YH, T, L - T, [3.02, 7.75]),
]
DOORS = {
    'puerta_entrada_living': (T / 2, 3.17),
    'puerta_salida_laundry': (T / 2, 7.50),
    'puerta_living_kitchen': (XV, 2.73),
    'puerta_laundry_bedroom': (XV, 8.06),
    'puerta_laundry_living': (3.02, YH),
    'puerta_bedroom_kitchen': (7.75, YH),
}

DINNER = (6.69, 3.05)
DINNER_R = 0.75
CHAIR_D = 0.72
ESTANTE_NIVELES = (0.45, 0.85, 1.25, 1.62)
WOOD = '0.45 0.30 0.18'
DARK = '0.25 0.22 0.20'
WHITE = '0.92 0.92 0.92'
GREY = '0.55 0.55 0.58'
GREEN = '0.20 0.50 0.20'
SOFA = '0.80 0.80 0.80'
FURN = [
    ('laundry_table', 'table', (1.43, 8.40), (1.60, 0.90), 0.75, DARK),
    ('washing_machine', 'box', (2.85, 8.55), (0.60, 0.60), 0.85, WHITE),
    ('shelf', 'estante', (1.00, 5.43), (1.20, 0.40), 2.00, WHITE, (0, 1)),
    ('laundry_trash_bin', 'cyl', (2.02, 5.45), 0.17, 0.60, GREY),
    ('bed', 'box', (5.55, 5.73), (2.00, 1.00), 0.50, WHITE),
    ('bedside_table', 'box', (5.15, 6.55), (0.50, 0.45), 0.55, DARK),
    ('coat_rack', 'cyl', (8.55, 8.50), 0.20, 1.70, WOOD),
    ('bookshelf', 'estante', (8.65, 7.00), (0.40, 1.00), 2.00, WOOD, (-1, 0)),
    ('planta_bedroom', 'cyl', (8.55, 5.50), 0.15, 0.70, GREEN),
    ('tv_stand', 'box', (1.42, 4.88), (1.30, 0.40), 0.50, WOOD),
    ('sofa', 'box', (2.25, 0.60), (2.10, 0.90), 0.80, SOFA),
    ('coffee_table', 'table', (3.95, 0.42), (0.60, 0.45), 0.45, DARK),
    ('planta_living', 'cyl', (0.60, 0.65), 0.15, 0.70, GREEN),
    ('cabinet', 'box', (4.75, 4.25), (0.40, 1.20), 0.90, WHITE),
    ('refrigerator', 'box', (8.525, 4.25), (0.65, 0.80), 1.80, GREY),
    ('kitchen_trash_bin', 'cyl', (8.55, 2.10), 0.18, 0.60, GREY),
    ('dinner_table', 'round', DINNER, DINNER_R, 0.75, WOOD),
    ('silla_n', 'chair', (DINNER[0], DINNER[1] + CHAIR_D), 0.42, 0.85, DARK, -math.pi / 2),
    ('silla_s', 'chair', (DINNER[0], DINNER[1] - CHAIR_D), 0.42, 0.85, DARK, math.pi / 2),
    ('silla_o', 'chair', (DINNER[0] - CHAIR_D, DINNER[1]), 0.42, 0.85, DARK, 0.0),
    ('silla_e', 'chair', (DINNER[0] + CHAIR_D, DINNER[1]), 0.42, 0.85, DARK, math.pi),
    ('counter', 'box', (5.10, 0.45), (0.75, 0.60), 0.90, DARK),
    ('sink', 'box', (5.80, 0.45), (0.60, 0.60), 0.90, GREY),
    ('cooking_table', 'box', (6.40, 0.45), (0.60, 0.60), 0.90, DARK),
    ('dishwasher', 'box', (7.60, 0.45), (0.60, 0.60), 0.85, WHITE),
]

S2 = math.sqrt(0.5)
LOCATIONS = [
    (1, 'laundry_table', (0, -1), 'laundry', True),
    (2, 'washing_machine', (0, -1), 'laundry', True),
    (3, 'shelf', (0, 1), 'laundry', True),
    (4, 'laundry_trash_bin', (0, 1), 'laundry', False),
    (5, 'bed', (1, 0), 'bedroom', True, 0.12),
    (17, 'bedside_table', (0, 1), 'bedroom', True),
    (19, 'bookshelf', (-1, 0), 'bedroom', True),
    (18, 'coat_rack', (-S2, -S2), 'bedroom', False),
    (6, 'tv_stand', (0, -1), 'living_room', True),
    (7, 'sofa', (0, 1), 'living_room', True),
    (8, 'coffee_table', (0, 1), 'living_room', True, 0.20),
    (9, 'cabinet', (1, 0), 'kitchen', True),
    (10, 'refrigerator', (-1, 0), 'kitchen', False),
    (11, 'counter', (0, 1), 'kitchen', True, -0.10),
    (12, 'sink', (0, 1), 'kitchen', False),
    (13, 'cooking_table', (0, 1), 'kitchen', True),
    (14, 'dishwasher', (0, 1), 'kitchen', True),
    (15, 'kitchen_trash_bin', (-1, 0), 'kitchen', False),
    (16, 'dinner_table', (S2, -S2), 'kitchen', True),
]
ROOMS = {
    'laundry': (2.30, 7.00, 0.0),
    'bedroom': (6.90, 7.30, 0.0),
    'living_room': (2.40, 2.80, 0.0),
    'kitchen': (5.90, 1.80, math.pi / 2),
    'entrada': (SPAWN_UV[0], SPAWN_UV[1], 0.0),
    'salida': (-0.9, 7.50, math.pi),
}

ALCANCE, PUNTA, D_NOMINAL, D_PISO = 0.97, 0.44, 0.30, 0.05
LIM_NOMINAL = ALCANCE - PUNTA - D_NOMINAL
LIM_PISO = ALCANCE - PUNTA - D_PISO
MARGEN = 0.03
DIST_PRE = 0.6
OBJETOS = [
    ('botella_laundry', 'botella', 'laundry_table', 0.30, 0.40, 0.0, None),
    ('taza_laundry', 'taza', 'laundry_table', 0.38, 0.00, 2.4, None),
    ('galletas_laundry', 'galletas', 'laundry_table', 0.42, -0.42, 0.3, None),
    ('taza_comedor', 'taza', 'dinner_table', 0.40, 0.00, -0.8, None),
    ('botella_comedor', 'botella', 'dinner_table', 0.32, 0.12, 0.0, None),
    ('galletas_comedor', 'galletas', 'dinner_table', 0.36, -0.15, 1.2, None),
    ('taza_estante', 'taza', 'shelf', 0.29, 0.25, 0.0, 1.25),
    ('botella_estante', 'botella', 'shelf', 0.32, -0.30, 0.0, 1.62),
    ('galletas_librero', 'galletas', 'bookshelf', 0.30, 0.20, math.pi / 2, 1.62),
    ('botella_librero', 'botella', 'bookshelf', 0.32, -0.25, 0.0, 0.85),
]
OBJETOS_SUELO = [
    ('botella_suelo', 'botella', 'sofa', 0.20, 0.75, 0.0),
    ('taza_suelo', 'taza', 'tv_stand', 0.20, 0.50, 1.0),
]


def escalar(e):
    global L, XV, YH, SPAWN_UV, DINNER, WALL_LINES, DOORS, FURN, ROOMS
    if e == 1.0:
        return
    caras = [T, XV - T / 2, XV + T / 2, L - T]
    caras_v = [T, YH - T / 2, YH + T / 2, L - T]
    nL, nXV, nYH = L * e, XV * e, YH * e
    nuevas_u = [T, nXV - T / 2, nXV + T / 2, nL - T]
    nuevas_v = [T, nYH - T / 2, nYH + T / 2, nL - T]

    def eje(c, h, viejas, nuevas):
        for f, nf in zip(viejas, nuevas):
            if abs((c - h) - f) <= 0.13:
                return nf + ((c - h) - f) + h
            if abs(f - (c + h)) <= 0.13:
                return nf - (f - (c + h)) - h
        return c * e

    nfurn = []
    for f in FURN:
        name, kind, (u, v), size = f[:4]
        if kind == 'chair':
            continue
        hu, hv = (size, size) if kind in ('cyl', 'round') else (size[0] / 2, size[1] / 2)
        if kind == 'round':
            nu, nv = u * e, v * e
        else:
            nu, nv = eje(u, hu, caras, nuevas_u), eje(v, hv, caras_v, nuevas_v)
        nfurn.append((name, kind, (round(nu, 3), round(nv, 3))) + tuple(f[3:]))
    DINNER = next(f[2] for f in nfurn if f[0] == 'dinner_table')
    dc = next(f for f in FURN if f[0] == 'silla_n')[3:6]
    for name, du, dv, yaw in (('silla_n', 0, 1, -math.pi / 2), ('silla_s', 0, -1, math.pi / 2),
                              ('silla_o', -1, 0, 0.0), ('silla_e', 1, 0, math.pi)):
        pos = (DINNER[0] + du * CHAIR_D, DINNER[1] + dv * CHAIR_D)
        nfurn.append((name, 'chair', pos) + tuple(dc) + (yaw,))
    FURN = nfurn

    def puerta(d):
        return d * e

    WALL_LINES = [
        ('h', T / 2, 0.0, nL, []),
        ('h', nL - T / 2, 0.0, nL, []),
        ('v', nL - T / 2, T, nL - T, []),
        ('v', T / 2, T, nL - T, [puerta(3.17), puerta(7.50)]),
        ('v', nXV, T, nL - T, [puerta(2.73), puerta(8.06)]),
        ('h', nYH, T, nL - T, [puerta(3.02), puerta(7.75)]),
    ]
    DOORS = {k: (u if u == T / 2 else u * e, v * e) for k, (u, v) in DOORS.items()}
    ROOMS = {k: (u * e if u > 0 else u, v * e, y) for k, (u, v, y) in ROOMS.items()}
    SPAWN_UV = (SPAWN_UV[0], SPAWN_UV[1] * e)
    L, XV, YH = nL, nXV, nYH


escalar(ESCALA)


def wall_prims():
    out = []
    for i, (ax, k, a, b, doors) in enumerate(WALL_LINES):
        cuts = sorted((d - DOOR / 2, d + DOOR / 2) for d in doors)
        segs, s = [], a
        for c0, c1 in cuts:
            segs.append((s, c0))
            s = c1
        segs.append((s, b))
        for j, (s0, s1) in enumerate(segs):
            if s1 - s0 < 1e-3:
                continue
            m, ln = (s0 + s1) / 2, s1 - s0
            if ax == 'h':
                out.append(
                    dict(
                        name=f'muro_{i}_{j}',
                        kind='box',
                        c=(m, k, WALL_H / 2),
                        size=(ln, T, WALL_H),
                        yaw=0.0,
                        color='0.85 0.83 0.78',
                    )
                )
            else:
                out.append(
                    dict(
                        name=f'muro_{i}_{j}',
                        kind='box',
                        c=(k, m, WALL_H / 2),
                        size=(T, ln, WALL_H),
                        yaw=0.0,
                        color='0.85 0.83 0.78',
                    )
                )
    return out


def faldon(kind, u, v, size, h, yaw=0.0):
    """Bloque sólido bajo la cubierta, al ras de su borde, de FALDON_Z[0] a FALDON_Z[1]
    (sin pasar la cubierta). Visto desde el lidar es lo mismo que un cartón alrededor."""
    z0, z1 = FALDON_Z[0], min(FALDON_Z[1], h - 0.05)
    if z1 <= z0:
        return []
    zc, hz = (z0 + z1) / 2, z1 - z0
    if kind == 'round':
        return [dict(kind='cyl', c=(u, v, zc), r=size, h=hz, yaw=0.0)]
    sx, sy = (size, size) if kind == 'chair' else size
    return [dict(kind='box', c=(u, v, zc), size=(sx, sy, hz), yaw=yaw)]


def furn_prims(f):
    name, kind, (u, v), size, h, color = f[:6]
    P = []
    if FALDONES and kind in ('table', 'round', 'chair'):
        alto = 0.45 if kind == 'chair' else h
        for p in faldon(kind, u, v, size, alto, f[6] if kind == 'chair' else 0.0):
            p['color'] = '0.80 0.68 0.50'
            P.append(p)
    if kind == 'estante':
        su, sv = size
        nu, nv = f[6]
        base = ESTANTE_NIVELES[0]
        tb = 0.02
        P.append(dict(kind='box', c=(u, v, base / 2), size=(su, sv, base), yaw=0.0))
        zc, hz = (base + h) / 2, h - base
        if nv:
            P.append(dict(kind='box', c=(u, v - nv * (sv / 2 - tb / 2), zc), size=(su, tb, hz),
                          yaw=0.0))
            for s_ in (-1, 1):
                P.append(dict(kind='box', c=(u + s_ * (su / 2 - tb / 2), v, zc),
                              size=(tb, sv, hz), yaw=0.0))
        else:
            P.append(dict(kind='box', c=(u - nu * (su / 2 - tb / 2), v, zc), size=(tb, sv, hz),
                          yaw=0.0))
            for s_ in (-1, 1):
                P.append(dict(kind='box', c=(u, v + s_ * (sv / 2 - tb / 2), zc),
                              size=(su, tb, hz), yaw=0.0))
        for z in ESTANTE_NIVELES[1:] + (h,):
            P.append(dict(kind='box', c=(u, v, z - tb / 2), size=(su, sv, tb), yaw=0.0))
    elif kind == 'box':
        P.append(dict(kind='box', c=(u, v, h / 2), size=(size[0], size[1], h), yaw=0.0))
    elif kind == 'cyl':
        P.append(dict(kind='cyl', c=(u, v, h / 2), r=size, h=h, yaw=0.0))
    elif kind == 'table':
        sx, sy = size
        P.append(dict(kind='box', c=(u, v, h - 0.02), size=(sx, sy, 0.04), yaw=0.0))
        lx, ly, lh = sx / 2 - 0.06, sy / 2 - 0.06, h - 0.04
        for dx in (-lx, lx):
            for dy in (-ly, ly):
                P.append(
                    dict(kind='box', c=(u + dx, v + dy, lh / 2), size=(0.05, 0.05, lh), yaw=0.0)
                )
    elif kind == 'round':
        r = size
        P.append(dict(kind='cyl', c=(u, v, h - 0.02), r=r, h=0.04, yaw=0.0))
        d, lh = 0.65 * r * S2 * 1.0, h - 0.04
        for dx in (-d, d):
            for dy in (-d, d):
                P.append(
                    dict(kind='box', c=(u + dx, v + dy, lh / 2), size=(0.05, 0.05, lh), yaw=0.0)
                )
    elif kind == 'chair':
        s, yaw = size, f[6]
        seat = 0.45
        cy, sy_ = math.cos(yaw), math.sin(yaw)

        def rot(dx, dy):
            return (u + dx * cy - dy * sy_, v + dx * sy_ + dy * cy)

        P.append(dict(kind='box', c=(u, v, seat - 0.02), size=(s, s, 0.04), yaw=yaw))
        lo = s / 2 - 0.03
        for dx in (-lo, lo):
            for dy in (-lo, lo):
                pu, pv = rot(dx, dy)
                P.append(
                    dict(
                        kind='box',
                        c=(pu, pv, (seat - 0.04) / 2),
                        size=(0.04, 0.04, seat - 0.04),
                        yaw=yaw,
                    )
                )
        bu, bv = rot(-s / 2 + 0.02, 0.0)
        P.append(
            dict(kind='box', c=(bu, bv, seat + (h - seat) / 2), size=(0.04, s, h - seat), yaw=yaw)
        )
    for p in P:
        p.setdefault('color', color)
    return P


def geom_xml(p):
    if p['kind'] == 'box':
        return '<box><size>%.3f %.3f %.3f</size></box>' % p['size']
    return '<cylinder><radius>%.3f</radius><length>%.3f</length></cylinder>' % (p['r'], p['h'])


def model_xml(name, prims):
    links = []
    for i, p in enumerate(prims):
        x, y = to_world(p['c'][0], p['c'][1])
        g = geom_xml(p)
        col = p['color']
        links.append(
            f'      <link name="l{i}">'
            f'<pose>{x:.3f} {y:.3f} {p["c"][2]:.3f} 0 0 {p["yaw"]:.4f}</pose>\n'
            f'        <collision name="c"><geometry>{g}</geometry></collision>\n'
            f'        <visual name="v"><geometry>{g}</geometry>'
            f'<material><ambient>{col} 1</ambient><diffuse>{col} 1</diffuse></material></visual>\n'
            f'      </link>'
        )
    return (
        f'    <model name="{name}">\n      <static>true</static>\n'
        + '\n'.join(links)
        + '\n    </model>'
    )


def floor_xml():
    cx, cy = to_world(L / 2, L / 2)
    return (
        f'    <model name="piso_casa">\n      <static>true</static>\n'
        f'      <link name="l"><pose>{cx:.3f} {cy:.3f} 0.0005 0 0 0</pose>\n'
        f'        <visual name="v"><geometry><box><size>{L} {L} 0.001</size></box></geometry>'
        f'<material><ambient>0.93 0.84 0.68 1</ambient>'
        f'<diffuse>0.93 0.84 0.68 1</diffuse></material></visual>\n'
        f'      </link>\n    </model>'
    )


NARANJA, ROJO, ROJO_T = '0.86 0.40 0.22', '0.80 0.10 0.12', '0.82 0.08 0.08'
MODELOS = {
    'botella': dict(
        masa=0.55,
        col=('cyl', (0.033, 0.22)),
        piezas=[
            ('cyl', (0, 0, 0.075), (0.033, 0.150), NARANJA, 0),
            ('cyl', (0, 0, 0.160), (0.027, 0.020), NARANJA, 0),
            ('cyl', (0, 0, 0.182), (0.015, 0.025), '0.75 0.78 0.80', 0.4),
            ('cyl', (0, 0, 0.192), (0.019, 0.004), '0.75 0.78 0.80', 0.2),
            ('cyl', (0, 0, 0.2075), (0.016, 0.025), ROJO, 0),
            ('cyl', (0, 0, 0.130), (0.034, 0.012), ROJO, 0),
            ('cyl', (0, 0, 0.0625), (0.034, 0.015), '0.85 0.75 0.20', 0),
            ('cyl', (0, 0, 0.0500), (0.034, 0.010), '0.30 0.60 0.25', 0),
            ('cyl', (0, 0, 0.0375), (0.034, 0.015), ROJO, 0),
        ],
    ),
    'taza': dict(
        masa=0.30,
        col=('cyl', (0.043, 0.095)),
        piezas=[
            ('cyl', (0, 0, 0.0475), (0.043, 0.095), ROJO_T, 0),
            ('cyl', (0, 0, 0.0945), (0.037, 0.002), '0.95 0.95 0.95', 0),
            ('box', (0.073, 0, 0.048), (0.012, 0.014, 0.066), ROJO_T, 0),
            ('box', (0.058, 0, 0.075), (0.032, 0.014, 0.012), ROJO_T, 0),
            ('box', (0.058, 0, 0.021), (0.032, 0.014, 0.012), ROJO_T, 0),
        ],
    ),
    'galletas': dict(
        masa=0.13,
        col=('box', (0.19, 0.062, 0.035)),
        piezas=[
            ('box', (0, 0, 0.0175), (0.19, 0.062, 0.035), '0.70 0.08 0.10', 0),
            ('box', (0.02, 0, 0.0355), (0.12, 0.030, 0.001), '0.85 0.65 0.30', 0),
        ],
    ),
}


def inercia(forma, m, med):
    """Momentos de inercia (ixx, iyy, izz) de un cilindro o una caja macizos."""
    if forma == 'cyl':
        r, h = med
        i = m * (3 * r * r + h * h) / 12
        return i, i, m * r * r / 2
    sx, sy, sz = med
    return m * (sy**2 + sz**2) / 12, m * (sx**2 + sz**2) / 12, m * (sx**2 + sy**2) / 12


def geom_obj(forma, med):
    if forma == 'cyl':
        return '<cylinder><radius>%.4f</radius><length>%.4f</length></cylinder>' % med
    return '<box><size>%.4f %.4f %.4f</size></box>' % med


def objeto_xml(o):
    m = MODELOS[o['modelo']]
    forma, med = m['col']
    zc = med[1] / 2 if forma == 'cyl' else med[2] / 2
    ixx, iyy, izz = inercia(forma, m['masa'], med)
    vis = []
    for k, (f, (px, py, pz), md, col, tr) in enumerate(m['piezas']):
        vis.append(
            f'        <visual name="v{k}"><pose>{px} {py} {pz} 0 0 0</pose>'
            f'<geometry>{geom_obj(f, md)}</geometry><transparency>{tr}</transparency>'
            f'<material><ambient>{col} 1</ambient><diffuse>{col} 1</diffuse></material></visual>'
        )
    return (
        f'    <model name="{o["nombre"]}">\n'
        f'      <pose>{o["x"]:.3f} {o["y"]:.3f} {o["z"]:.3f} 0 0 {o["yaw"]:.3f}</pose>\n'
        f'      <link name="link">\n'
        f'        <inertial><pose>0 0 {zc:.4f} 0 0 0</pose><mass>{m["masa"]}</mass>'
        f'<inertia><ixx>{ixx:.3e}</ixx><iyy>{iyy:.3e}</iyy><izz>{izz:.3e}</izz>'
        f'<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>\n'
        f'        <collision name="c"><pose>0 0 {zc:.4f} 0 0 0</pose>'
        f'<geometry>{geom_obj(forma, med)}</geometry>'
        f'<surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>'
        f'</collision>\n' + '\n'.join(vis) + '\n      </link>\n    </model>'
    )


def medio_ancho(o, ang):
    """Medio ancho del objeto (su colisión) medido en la dirección 'ang' (rad, en el plano)."""
    forma, med = MODELOS[o['modelo']]['col']
    if forma == 'cyl':
        return med[0] + (0.036 if o['modelo'] == 'taza' else 0.0)
    a = ang - o['yaw']
    return abs(math.cos(a)) * med[0] / 2 + abs(math.sin(a)) * med[1] / 2


def ubicar_objetos(by_name):
    """Pasa OBJETOS (mueble, prof, lateral, nivel) y OBJETOS_SUELO a coordenadas del mundo y
    calcula la profundidad EFECTIVA (desde el borde más cercano al que Milo podría llegar)."""
    out = []
    lado = {loc[1]: loc for loc in LOCATIONS}
    filas = [o + ('mueble',) for o in OBJETOS]
    filas += [(n, m, mb, -sep, lat, yaw, 0.0, 'suelo')
              for n, m, mb, sep, lat, yaw in OBJETOS_SUELO]
    for nombre, modelo, mueble, prof, lat, yaw, nivel, donde in filas:
        f = by_name[mueble]
        kind, (cu, cv), size, h = f[1], f[2], f[3], f[4]
        nu, nv = lado[mueble][2]
        if kind in ('round', 'cyl'):
            half = size
        else:
            half = abs(nu) * size[0] / 2 + abs(nv) * size[1] / 2
        tu, tv = nv, -nu
        u = cu + nu * (half - prof) + tu * lat
        v = cv + nv * (half - prof) + tv * lat
        yaw_w = math.atan2(-nv, -nu) + yaw
        yaw_w = math.atan2(math.sin(yaw_w), math.cos(yaw_w))
        z = nivel if nivel is not None else h
        o = dict(nombre=nombre, modelo=modelo, mueble=mueble, donde=donde, prof=prof, lateral=lat,
                 u=u, v=v, yaw=yaw_w, z=round(z + 0.001, 3))
        o['x'], o['y'] = to_world(u, v)
        if donde == 'mueble':
            d = half + DIST_PRE + PUNTA
            o['pre'] = (*to_world(cu + nu * d + tu * lat, cv + nv * d + tv * lat),
                        math.atan2(-nv, -nu))
        if donde == 'suelo':
            o['prof_ef'], o['cabe'] = None, True
        elif kind == 'round':
            rho = math.hypot(u - cu, v - cv)
            o['prof_ef'] = size - rho
            o['cabe'] = rho + medio_ancho(o, math.atan2(v - cv, u - cu)) <= size
        elif kind == 'estante':
            o['prof_ef'] = prof
            fondo = 2 * half - 0.02
            ancho = abs(nv) * size[0] + abs(nu) * size[1] - 0.04
            ang_in = math.atan2(-nv, -nu)
            o['cabe'] = (prof + medio_ancho(o, ang_in) <= fondo
                         and abs(lat) + medio_ancho(o, ang_in + math.pi / 2) <= ancho / 2)
        else:
            hu, hv = size[0] / 2, size[1] / 2
            du, dv = u - cu, v - cv
            o['prof_ef'] = min(hu - abs(du), hv - abs(dv))
            o['cabe'] = (abs(du) + medio_ancho(o, 0.0) <= hu
                         and abs(dv) + medio_ancho(o, math.pi / 2) <= hv)
        out.append(o)
    return out


def objeto_prim(o):
    """Colisión del objeto como primitiva (coordenadas casa) para la verificación."""
    forma, med = MODELOS[o['modelo']]['col']
    if forma == 'cyl':
        return dict(kind='cyl', c=(o['u'], o['v'], o['z'] + med[1] / 2), r=med[0], h=med[1],
                    yaw=0.0)
    return dict(kind='box', c=(o['u'], o['v'], o['z'] + med[2] / 2), size=med, yaw=o['yaw'])


def write_objetos_yaml(objs):
    lineas = [
        '# GENERADO por andesrobot_gazebo/tools/gen_casa_robocup.py — NO editar a mano.',
        '# Objetos de robocup_casa.world: pose en el mundo de Gazebo (= frame map en la sim), en',
        '# la base del objeto. Hace de "módulo de percepción": en el paper el robot recibe esta',
        '# pose + ruido.',
        '#   donde: mueble (mesa o estante) | suelo (por debajo del lidar: no lo ve)',
        '#   prof_ef: profundidad EFECTIVA, centro del objeto medido desde el borde más cercano',
        f'#            del mueble (alcance {ALCANCE} m: <= {LIM_NOMINAL:.2f} con el filtro',
        f'#            nominal, <= {LIM_PISO:.2f} solo relajando hasta el piso duro)',
        '#   sep: en el suelo, distancia desde la cara del mueble de al lado',
        '#   pre: [x, y, yaw] de base_footprint en el pre-agarre (punta del robot a 0.6 m del',
        '#        borde, mirando al mueble); desde ahí el robot orienta y se acerca al objeto',
    ]
    for o in objs:
        extra = (f'prof_ef: {o["prof_ef"]:.2f}' if o['donde'] != 'suelo'
                 else f'sep: {-o["prof"]:.2f}')
        if 'pre' in o:
            extra += f', pre: [{o["pre"][0]:.3f}, {o["pre"][1]:.3f}, {o["pre"][2]:.3f}]'
        lineas.append(
            f'{o["nombre"]}: {{modelo: {o["modelo"]}, mueble: {o["mueble"]}, '
            f'donde: {o["donde"]}, x: {o["x"]:.3f}, y: {o["y"]:.3f}, z: {o["z"]:.3f}, '
            f'yaw: {o["yaw"]:.3f}, {extra}}}'
        )
    with open(OUT_OBJ, 'w') as fh:
        fh.write('\n'.join(lineas) + '\n')


def write_world(walls, furn, objs=()):
    cam_x, cam_y = to_world(L / 2, -0.62 * L)
    body = [model_xml('muros', walls), floor_xml()]
    body += [model_xml(name, prims) for name, prims in furn]
    body += [objeto_xml(o) for o in objs]
    xml = f'''<?xml version="1.0"?>
<!-- GENERADO por andesrobot_gazebo/tools/gen_casa_robocup.py — NO editar a mano.
     Casa RoboCup@Home (rulebook): Laundry, Bedroom, Living room, Kitchen. {L:.1f} x {L:.1f} m,
     puertas de {DOOR:.2f} m. Origen del mundo (0,0) = CENTRO de la casa; Milo parte afuera de la
     puerta del living (CASA_SPAWN en sim.sh). Mesas y sillas con faldón de cartón a la altura del
     lidar ({LIDAR_Z} m), así el lidar ve su contorno. Objetos para agarrar: ver
     robocup_casa_objetos.yaml. -->
<sdf version="1.6">
  <world name="robocup_casa">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <scene><shadows>false</shadows></scene>
    <gui><camera name="user_camera">
      <pose>{cam_x:.2f} {cam_y:.2f} {1.45 * L:.1f} 0 0.95 1.5708</pose></camera></gui>
{chr(10).join(body)}
  </world>
</sdf>
'''
    os.makedirs(os.path.dirname(OUT_WORLD), exist_ok=True)
    with open(OUT_WORLD, 'w') as fh:
        fh.write(xml)


RES = 0.02
U0, V0 = -1.5, -0.5
NU, NV = int((L + 2.0 - U0) / RES), int((L + 0.5 - V0) / RES)
UU, VV = np.meshgrid(U0 + (np.arange(NU) + 0.5) * RES, V0 + (np.arange(NV) + 0.5) * RES)


def raster(prims, zslice=None):
    """Ocupación 2D. zslice=None -> todo lo que exista entre 0 y 1.3 m (choque físico);
    zslice=z -> solo lo que corta el plano z (lo que ve un lidar a esa altura)."""
    occ = np.zeros((NV, NU), bool)
    for p in prims:
        cz = p['c'][2]
        hz = (p['size'][2] if p['kind'] == 'box' else p['h']) / 2
        if zslice is not None and not (cz - hz <= zslice <= cz + hz):
            continue
        if zslice is None and cz - hz > 1.3:
            continue
        du, dv = UU - p['c'][0], VV - p['c'][1]
        if p['kind'] == 'cyl':
            occ |= du * du + dv * dv <= p['r'] ** 2
        else:
            c, s = math.cos(-p['yaw']), math.sin(-p['yaw'])
            lu, lv = du * c - dv * s, du * s + dv * c
            occ |= (np.abs(lu) <= p['size'][0] / 2) & (np.abs(lv) <= p['size'][1] / 2)
    return occ


def dist_field(occ):
    """Distancia (m) a la celda ocupada más cercana (exacta con scipy; si no, chamfer aprox.)."""
    try:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(~occ) * RES
    except ImportError:
        pass
    INF = 1e9
    d = np.where(occ, 0.0, INF)
    a, b = RES, RES * math.sqrt(2)
    for _ in range(2):
        for i in range(1, NV):
            d[i] = np.minimum(
                d[i],
                np.minimum.reduce(
                    [d[i - 1] + a, np.r_[INF, d[i - 1][:-1]] + b, np.r_[d[i - 1][1:], INF] + b]
                ),
            )
            row = d[i]
            for j in range(1, NU):
                if row[j - 1] + a < row[j]:
                    row[j] = row[j - 1] + a
        for i in range(NV - 2, -1, -1):
            d[i] = np.minimum(
                d[i],
                np.minimum.reduce(
                    [d[i + 1] + a, np.r_[INF, d[i + 1][:-1]] + b, np.r_[d[i + 1][1:], INF] + b]
                ),
            )
            row = d[i]
            for j in range(NU - 2, -1, -1):
                if row[j + 1] + a < row[j]:
                    row[j] = row[j + 1] + a
    return d


def idx(u, v):
    return int((v - V0) / RES), int((u - U0) / RES)


def footprint_cells(u, v, yaw_house):
    c, s = math.cos(yaw_house), math.sin(yaw_house)
    du, dv = UU - u, VV - v
    lx, ly = du * c + dv * s, -du * s + dv * c
    return (lx >= FP[2][0]) & (lx <= FP[0][0]) & (np.abs(ly) <= FP[0][1])


def main():
    walls = wall_prims()
    furn = [(f[0], furn_prims(f)) for f in FURN]
    allp = walls + [p for _, ps in furn for p in ps]
    by_name = {f[0]: f for f in FURN}
    objs = ubicar_objetos(by_name)
    allp += [objeto_prim(o) for o in objs]
    write_world(walls, furn, objs)
    write_objetos_yaml(objs)

    goals = []
    for loc in LOCATIONS:
        rid, fname, (nu, nv), room, place = loc[:5]
        shift = loc[5] if len(loc) > 5 else 0.0
        f = by_name[fname]
        (cu, cv), size = f[2], f[3]
        if f[1] in ('cyl', 'round'):
            half = size
        else:
            half = abs(nu) * size[0] / 2 + abs(nv) * size[1] / 2
            if nu and nv:
                half = min(size[0], size[1]) / 2
        gu = cu + nu * (half + APPROACH) - nv * shift
        gv = cv + nv * (half + APPROACH) + nu * shift
        yaw = math.atan2(-nv, -nu)
        goals.append(dict(name=fname, id=rid, room=room, place=place, u=gu, v=gv, yaw=yaw))
    for rname, (ru, rv, ryaw) in ROOMS.items():
        goals.append(dict(name=rname, id=None, room=rname, place=False, u=ru, v=rv, yaw=ryaw))

    occ_phys = raster(allp)
    occ_lidar = raster(allp, LIDAR_Z)
    d_phys = dist_field(occ_phys)
    ok = True
    for g in goals:
        fp = footprint_cells(g['u'], g['v'], g['yaw'])
        clear = float(d_phys[fp].min())
        g['clear'] = clear
        if clear < 0.03:
            print(f'  [MAL] {g["name"]}: el footprint choca (holgura {clear:.2f} m)')
            ok = False
        c, s = math.cos(g['yaw']), math.sin(g['yaw'])
        du, dv = UU - g['u'], VV - g['v']
        lx, ly = du * c + dv * s, -du * s + dv * c
        stop = (lx > FP[0][0]) & (lx <= FP[0][0] + 0.30) & (np.abs(ly) <= FP[0][1] + 0.10)
        if occ_lidar[stop].any():
            print(f'  [OJO] {g["name"]}: hay algo dentro de la zona de parada del safety filter')
            ok = False
        if occ_lidar[du * du + dv * dv < TURN_R**2].any():
            print(f'  [OJO] {g["name"]}: no podría girar en el lugar (algo a < {TURN_R:.2f} m)')
            ok = False
    free = d_phys > 0.27
    start = idx(*SPAWN_UV)
    seen = np.zeros_like(free)
    dq = deque([start])
    seen[start] = True
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < NV and 0 <= b < NU and free[a, b] and not seen[a, b]:
                seen[a, b] = True
                dq.append((a, b))
    for g in goals:
        g['reach'] = bool(seen[idx(g['u'], g['v'])])
        if not g['reach']:
            print(f'  [MAL] {g["name"]}: no se llega desde la partida')
            ok = False
    for dn, (du_, dv_) in DOORS.items():
        print(f'  puerta {dn:24s} libre a cada lado del centro: {d_phys[idx(du_, dv_)]:.2f} m')

    for o in objs:
        if o['donde'] == 'suelo':
            print(f'  objeto {o["nombre"]:18s} en el suelo junto a {o["mueble"]:13s} '
                  f'(a {-o["prof"]:.2f} m de su cara; el lidar no lo ve)')
            continue
        pe = o['prof_ef']
        zona = ('nominal' if pe <= LIM_NOMINAL else
                'relajando' if pe <= LIM_PISO else 'no alcanza')
        en_franja = LIM_NOMINAL + MARGEN <= pe <= LIM_PISO - MARGEN
        nota = '' if o['cabe'] else '  [MAL] no cabe en la superficie'
        if not en_franja:
            nota += f'  [OJO] fuera de la franja {LIM_NOMINAL + MARGEN:.2f}..' \
                    f'{LIM_PISO - MARGEN:.2f}'
        print(f'  objeto {o["nombre"]:18s} en {o["mueble"]:13s} a {o["z"]:.2f} m de alto, '
              f'prof efectiva {pe:.2f} m ({zona}){nota}')
        ok &= o['cabe'] and en_franja

    try:
        import yaml

        with open(LUGARES) as fh:
            lug = yaml.safe_load(fh) or {}
        lug = lug.get('lugares', lug)
        for nombre, p in lug.items():
            u, v = p['x'] + L / 2, p['y'] + L / 2
            fp = footprint_cells(u, v, p['yaw'])
            clear = float(d_phys[fp].min())
            c, s = math.cos(p['yaw']), math.sin(p['yaw'])
            du, dv = UU - u, VV - v
            lx, ly = du * c + dv * s, -du * s + dv * c
            stop = (lx > FP[0][0]) & (lx <= FP[0][0] + 0.30) & (np.abs(ly) <= FP[0][1] + 0.10)
            nota = '' if clear >= 0.03 else '  [MAL] choca: vuelve a anotarlo'
            if not nota and occ_lidar[stop].any():
                nota = '  [OJO] algo dentro de la zona de parada del filtro'
            print(f'  tu lugar {nombre:14s} holgura {clear:.2f} m{nota}')
    except (ImportError, OSError, KeyError, TypeError) as e:
        print(f'  (no revisé {os.path.basename(LUGARES)}: {e})')

    try:
        from PIL import Image, ImageDraw

        sc = 4
        img = np.full((NV, NU, 3), 255, np.uint8)
        house = (UU >= 0) & (UU <= L) & (VV >= 0) & (VV <= L)
        img[house] = (238, 216, 176)
        img[seen & ~occ_phys] = (225, 240, 225)
        img[occ_phys] = (150, 150, 150)
        img[occ_lidar] = (30, 30, 30)
        im = Image.fromarray(img[::-1]).resize((NU * sc // 2, NV * sc // 2), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        k = sc / 2 / RES

        def px(u, v):
            return ((u - U0) * k, (V0 + NV * RES - v) * k)

        for rid, fname, *_ in LOCATIONS:
            (fu, fv) = next(f[2] for f in FURN if f[0] == fname)
            x0, y0 = px(fu, fv)
            dr.ellipse([x0 - 11, y0 - 9, x0 + 11, y0 + 9], fill=(70, 120, 230))
            dr.text((x0 - 6, y0 - 6), str(rid), fill=(255, 255, 255))
        for rname, (ru, rv, _) in ROOMS.items():
            if rname not in ('entrada', 'salida'):
                rx, ry = px(ru, rv)
                dr.text((rx - 20, ry), rname, fill=(120, 60, 20))
        colores = {'botella': (235, 110, 40), 'taza': (210, 20, 20), 'galletas': (150, 20, 60)}
        for o in objs:
            ox, oy = px(o['u'], o['v'])
            dr.ellipse([ox - 5, oy - 5, ox + 5, oy + 5], fill=colores[o['modelo']],
                       outline=(0, 0, 0))
        ex, ey = px(*SPAWN_UV)
        dr.ellipse([ex - 8, ey - 8, ex + 8, ey + 8], outline=(200, 0, 0), width=3)
        dr.text(
            (8, 8),
            'gris = choque fisico | negro = lo que ve el lidar (0.36 m) | '
            'verde = donde cabe Milo | azul = numero del rulebook | o rojo = partida | '
            'puntos = objetos (naranja botella, rojo taza, vino galletas)',
            fill=(0, 0, 0),
        )
        im.save(OUT_PNG)
    except ImportError:
        print('  (sin pillow: no se generó el PNG)')

    print(f'Mundo:       {OUT_WORLD}')
    print(f'Vista:       {OUT_PNG}')
    print(f'Objetos:     {OUT_OBJ}')
    sx, sy = to_world(*SPAWN_UV)
    print(f'Partida de Milo (afuera de la puerta del living): x:={sx} y:={sy} yaw:=0.0'
          '   <- tiene que coincidir con CASA_SPAWN en sim.sh')
    print('Verificación:', 'OK' if ok else 'HAY PROBLEMAS (ver arriba)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
