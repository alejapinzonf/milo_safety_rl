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
# Dónde quedó cada objeto (la "verdad" del simulador; hace de módulo de percepción)
OUT_OBJ = os.path.join(PKG_GAZEBO, 'worlds', 'robocup_casa_objetos.yaml')
# Lugares que anotaste a mano (tf2_echo): el script revisa que sigan libres con las mesas nuevas
LUGARES = os.path.join(SRC, 'andesrobot_slam', 'maps', 'casa_lugares.yaml')

# ------------------------------------------------------------------------------------------
# Parámetros generales
# ------------------------------------------------------------------------------------------
# ESCALA agranda la casa (distancias entre muros y muebles) pero NO los muebles ni el grosor de
# los muros: una cama sigue midiendo 2 m. Todas las medidas de más abajo son las del PLANO del
# rulebook (escala 1, casa de 9 x 9 m); escalar() las convierte al final de esta sección.
# Con 1.4 la casa queda de 12.6 x 12.6 m: Milo pasa con holgura y puede girar entre los muebles.
ESCALA = 1.4
L = 9.0  # la casa es de L x L m (medida exterior, en el plano)
T = 0.15  # grosor de muro
WALL_H = 1.2  # alto de muro (el lidar escanea a ~0.36 m)
# Ancho de puerta. Con 1.40 Milo puede GIRAR dentro de la puerta: el filtro de seguridad (modo
# teleop) bloquea el giro si hay algo a menos de 0.51 (radio de giro) + 0.15 = 0.66 m del centro,
# y en una puerta de 1.40 los marcos quedan a 0.70 m. Una puerta real de casa mide ~0.90.
DOOR = 1.40
SPAWN_UV = (-0.9, 3.17)  # punto de partida (afuera de la puerta del living), en coordenadas casa
# Altura del plano que barre el lidar (base_footprint -> frame "laser" del URDF). OJO: 0.333 m es
# la TAPA del chasis, no el lidar; el lidar va encima y barre a 0.363 m.
LIDAR_Z = 0.363
# FALDONES: "cartón" alrededor de mesas y sillas a la altura del lidar. Sin faldón el lidar (que
# barre a 0.36 m) ve solo las PATAS; con faldón ve el contorno completo de la cubierta, así el
# mapa sale con la mesa sólida, Nav2 no planea por debajo y el filtro de seguridad mide la
# distancia al BORDE de la mesa. En el robot real (RoboCup no deja tocar los muebles) el
# equivalente es dibujar el contorno en el mapa (zonas prohibidas de Nav2).
FALDONES = True
FALDON_Z = (0.15, 0.45)  # [m] desde dónde hasta dónde llega el faldón (cubre la altura del lidar)
APPROACH = 0.80  # distancia base_footprint -> cara del mueble en las poses objetivo
# (la punta de Milo está 0.44 m adelante => queda a ~0.36 m: más que el
#  stop_distance 0.30 del filtro de seguridad)
# Footprint de Milo en base_footprint (igual que safety.yaml / nav2_params.yaml)
FP = [(0.44, 0.25), (0.44, -0.25), (-0.12, -0.25), (-0.12, 0.25)]
# Girar en el lugar en la pose objetivo: el filtro (modo Nav2, safety_nav.yaml) lo bloquea si el
# lidar ve algo a menos de radio_de_giro + rotation_clearance.
TURN_R = math.hypot(0.44, 0.25) + 0.08

# Muros internos: centros
XV = 4.475  # muro vertical del medio (separa izquierda / derecha)
YH = 5.155  # muro horizontal del medio (separa arriba / abajo)


def to_world(u, v):
    # El centro de la casa es el (0, 0) del mundo de Gazebo.
    return (round(u - L / 2, 3), round(v - L / 2, 3))


# ------------------------------------------------------------------------------------------
# Muros con puertas: (eje, coordenada fija, desde, hasta, [centros de puertas])
# ------------------------------------------------------------------------------------------
WALL_LINES = [
    ('h', T / 2, 0.0, L, []),  # abajo
    ('h', L - T / 2, 0.0, L, []),  # arriba
    ('v', L - T / 2, T, L - T, []),  # derecha
    ('v', T / 2, T, L - T, [3.17, 7.50]),  # izquierda: puertas al living y al laundry
    ('v', XV, T, L - T, [2.73, 8.06]),  # medio: living<->kitchen, laundry<->bedroom
    ('h', YH, T, L - T, [3.02, 7.75]),  # medio: laundry<->living, bedroom<->kitchen
]
DOORS = {  # nombre de cada puerta (centro en coordenadas casa) -> para el .yaml y el paper
    'puerta_entrada_living': (T / 2, 3.17),
    'puerta_salida_laundry': (T / 2, 7.50),
    'puerta_living_kitchen': (XV, 2.73),
    'puerta_laundry_bedroom': (XV, 8.06),
    'puerta_laundry_living': (3.02, YH),
    'puerta_bedroom_kitchen': (7.75, YH),
}

# ------------------------------------------------------------------------------------------
# Muebles. tipo: box (sólido hasta el suelo), table (cubierta + 4 patas), cyl (cilindro sólido),
# round (mesa redonda con 4 patas), chair (asiento + patas + respaldo, mirando a "yaw"),
# estante (mueble cerrado abajo + repisas abiertas arriba, con el frente hacia "frente").
# Campos: nombre, tipo, (u, v), tamaño, alto, color, [extra]
#   box/table/estante: tamaño = (largo_u, ancho_v); cyl/round: tamaño = radio; chair: lado
#   extra: chair -> yaw; estante -> frente (du, dv) = hacia dónde mira el lado abierto
# ------------------------------------------------------------------------------------------
# Mesa del comedor: en el plano está en (6.69, 2.79); la subimos a 3.05 y la basurera de la cocina
# de 3.30 a 2.10 para que Milo (radio de giro 0.51 m) pueda girar frente a la cocina y la basurera.
DINNER = (6.69, 3.05)
# MESAS GRANDES (para el paper): tienen que ser más hondas que lo que alcanza el brazo (ver
# ALCANCE más abajo) para que haya objetos "difíciles" y otros que no se alcanzan desde ese lado.
DINNER_R = 0.75  # radio de la mesa redonda del comedor (antes 0.60)
CHAIR_D = 0.72  # centro de la mesa -> centro de cada silla (metidas bajo la cubierta)
# ESTANTE: abajo un mueble CERRADO hasta la primera repisa (el lidar ve su frente, como un
# mueble real con puertas abajo) y arriba repisas abiertas con fondo y costados. Las alturas son
# las de la SUPERFICIE de cada repisa; la más alta (1.62 m) está casi a la altura máxima del hombro
# del brazo (1.61 m), así que se alcanza con el brazo horizontal.
ESTANTE_NIVELES = (0.45, 0.85, 1.25, 1.62)
WOOD = '0.45 0.30 0.18'
DARK = '0.25 0.22 0.20'
WHITE = '0.92 0.92 0.92'
GREY = '0.55 0.55 0.58'
GREEN = '0.20 0.50 0.20'
SOFA = '0.80 0.80 0.80'
FURN = [
    # Laundry
    # mesa grande (antes 1.40 x 0.70), sigue pegada al muro de arriba (cara a 8.85)
    ('laundry_table', 'table', (1.43, 8.40), (1.60, 0.90), 0.75, DARK),
    ('washing_machine', 'box', (2.85, 8.55), (0.60, 0.60), 0.85, WHITE),
    # estante con repisas (antes una caja), pegado al muro del medio, abierto hacia +v
    ('shelf', 'estante', (1.00, 5.43), (1.20, 0.40), 2.00, WHITE, (0, 1)),
    ('laundry_trash_bin', 'cyl', (2.02, 5.45), 0.17, 0.60, GREY),
    # Bedroom
    ('bed', 'box', (5.55, 5.73), (2.00, 1.00), 0.50, WHITE),
    ('bedside_table', 'box', (5.15, 6.55), (0.50, 0.45), 0.55, DARK),
    ('coat_rack', 'cyl', (8.55, 8.50), 0.20, 1.70, WOOD),
    # librero nuevo (no está en el rulebook), pegado al muro derecho, abierto hacia -u
    ('bookshelf', 'estante', (8.65, 7.00), (0.40, 1.00), 2.00, WOOD, (-1, 0)),
    ('planta_bedroom', 'cyl', (8.55, 5.50), 0.15, 0.70, GREEN),
    # Living room
    ('tv_stand', 'box', (1.42, 4.88), (1.30, 0.40), 0.50, WOOD),
    ('sofa', 'box', (2.25, 0.60), (2.10, 0.90), 0.80, SOFA),
    ('coffee_table', 'table', (3.95, 0.42), (0.60, 0.45), 0.45, DARK),
    ('planta_living', 'cyl', (0.60, 0.65), 0.15, 0.70, GREEN),
    # Kitchen
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

# ------------------------------------------------------------------------------------------
# Ubicaciones con nombre (las del rulebook) -> desde qué lado se acerca Milo.
# (id_rulebook, mueble, dirección desde el mueble hacia Milo (du, dv), sala,
#  se_pueden_dejar_objetos
#  [, corrimiento lateral en m a lo largo de la cara, para alejarse de un muro y poder girar])
# ------------------------------------------------------------------------------------------
S2 = math.sqrt(0.5)
LOCATIONS = [
    (1, 'laundry_table', (0, -1), 'laundry', True),
    (2, 'washing_machine', (0, -1), 'laundry', True),
    (3, 'shelf', (0, 1), 'laundry', True),
    (4, 'laundry_trash_bin', (0, 1), 'laundry', False),
    (5, 'bed', (1, 0), 'bedroom', True, 0.12),
    (17, 'bedside_table', (0, 1), 'bedroom', True),
    (19, 'bookshelf', (-1, 0), 'bedroom', True),  # 19: no es del rulebook (librero agregado)
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
# Centros de sala + entrada/salida (u, v, yaw_mundo)
ROOMS = {
    'laundry': (2.30, 7.00, 0.0),
    'bedroom': (6.90, 7.30, 0.0),
    'living_room': (2.40, 2.80, 0.0),
    'kitchen': (5.90, 1.80, math.pi / 2),
    'entrada': (SPAWN_UV[0], SPAWN_UV[1], 0.0),
    'salida': (-0.9, 7.50, math.pi),
}

# ------------------------------------------------------------------------------------------
# Objetos para agarrar (paper: filtro de seguridad adaptativo).
#
# ALCANCE: hasta dónde llega la pinza al frente, medido desde base_footprint. Con el lift libre el
# hombro se pone a la altura del objeto y el brazo estira su radio completo: 0.1225 (eje de
# joint_1) + 0.848 = 0.97 m (informe del brazo, tabla de espacio de trabajo). Vale para objetos
# entre ~0.6 y ~1.9 m de alto (el hombro sube de 0.61 a 1.61 m). OJO: la tabla "a 0.75 m de alto:
# 0.89 m" del informe sale de un muestreo aleatorio que subestima el máximo; no usarla.
# Es un máximo geométrico (brazo estirado, sin choques): agarrar en serio pide unos cm menos.
# Con la punta de Milo en x = 0.44, la PROFUNDIDAD (cuánto adentro del mueble está el centro del
# objeto) decide:
#   profundidad <= LIM_NOMINAL (0.23)  se alcanza con el filtro nominal (para a 0.30 m del borde)
#   <= LIM_PISO (0.48)                 solo relajando el filtro hasta el piso duro (0.05 m)
#   más                                no se alcanza
# La profundidad que cuenta es la EFECTIVA: desde el borde MÁS CERCANO al que Milo podría llegar,
# no solo desde el lado de LOCATIONS. Si no, Milo "hace trampa" dándole la vuelta a la mesa.
#   mesa rectangular: el borde más cercano de los 4 (también el pegado al muro: así es
#                     conservador, nunca dice "difícil" a algo que es fácil desde otro lado)
#   mesa redonda:     radio - distancia del objeto al centro
#   estante:          solo el frente (los lados y el fondo son tablas)
# Todos los objetos de mesas y estantes van en la franja del medio, con 3 cm de margen.
ALCANCE, PUNTA, D_NOMINAL, D_PISO = 0.97, 0.44, 0.30, 0.05
LIM_NOMINAL = ALCANCE - PUNTA - D_NOMINAL  # 0.23 m
LIM_PISO = ALCANCE - PUNTA - D_PISO  # 0.48 m
MARGEN = 0.03
DIST_PRE = 0.6  # [m] del borde del mueble a la punta del robot en el pre-agarre (igual que rl/)
#
# Cada objeto se ubica respecto a la cara por donde se acerca Milo (la de LOCATIONS):
#   prof    = cuánto adentro del mueble, desde esa cara, en la dirección en que llega Milo
#   lateral = corrimiento a lo largo de la cara (+ = a la izquierda de Milo mirando el mueble)
#   yaw     = giro propio del objeto respecto a hacia dónde mira Milo (0 = su eje x apunta
#             hacia adentro del mueble)
#   nivel   = altura de la superficie donde va (None = encima de la mesa; en un estante, una
#             de ESTANTE_NIVELES)
# Campos: nombre, modelo, mueble, prof, lateral, yaw, nivel
# ------------------------------------------------------------------------------------------
OBJETOS = [
    # mesa de la lavandería (1.60 x 0.90, pegada al muro)
    ('botella_laundry', 'botella', 'laundry_table', 0.30, 0.40, 0.0, None),
    ('taza_laundry', 'taza', 'laundry_table', 0.38, 0.00, 2.4, None),
    ('galletas_laundry', 'galletas', 'laundry_table', 0.42, -0.42, 0.3, None),
    # mesa redonda del comedor (radio 0.75)
    ('taza_comedor', 'taza', 'dinner_table', 0.40, 0.00, -0.8, None),
    ('botella_comedor', 'botella', 'dinner_table', 0.32, 0.12, 0.0, None),
    ('galletas_comedor', 'galletas', 'dinner_table', 0.36, -0.15, 1.2, None),
    # estante de la lavandería y librero del dormitorio: objetos ALTOS (hasta 1.62 m)
    ('taza_estante', 'taza', 'shelf', 0.29, 0.25, 0.0, 1.25),
    ('botella_estante', 'botella', 'shelf', 0.32, -0.30, 0.0, 1.62),
    ('galletas_librero', 'galletas', 'bookshelf', 0.30, 0.20, math.pi / 2, 1.62),
    ('botella_librero', 'botella', 'bookshelf', 0.32, -0.25, 0.0, 0.85),
]
# Objetos en el SUELO, al lado de un mueble (afuera de él). Quedan por DEBAJO del plano del lidar
# (miden menos de 0.36 m): el lidar no los ve. Para agarrarlos el brazo tiene que bajar delante
# del robot y TAPA el lidar: es el caso del punto ciego del brazo del paper.
#   sep = distancia desde la cara del mueble (la de LOCATIONS) hasta el centro del objeto
# Campos: nombre, modelo, mueble, sep, lateral, yaw
OBJETOS_SUELO = [
    ('botella_suelo', 'botella', 'sofa', 0.20, 0.75, 0.0),
    ('taza_suelo', 'taza', 'tv_stand', 0.20, 0.50, 1.0),
]


# ------------------------------------------------------------------------------------------
# Escalar el plano. Los muebles pegados a un muro (a menos de 13 cm) se quedan pegados a ese
# muro; los demás se mueven proporcionalmente. Los tamaños no cambian.
# ------------------------------------------------------------------------------------------
def escalar(e):
    global L, XV, YH, SPAWN_UV, DINNER, WALL_LINES, DOORS, FURN, ROOMS
    if e == 1.0:
        return
    caras = [T, XV - T / 2, XV + T / 2, L - T]  # mismas caras de muro en u y en v (ver abajo)
    caras_v = [T, YH - T / 2, YH + T / 2, L - T]
    nL, nXV, nYH = L * e, XV * e, YH * e
    nuevas_u = [T, nXV - T / 2, nXV + T / 2, nL - T]
    nuevas_v = [T, nYH - T / 2, nYH + T / 2, nL - T]

    def eje(c, h, viejas, nuevas):
        for f, nf in zip(viejas, nuevas):
            if abs((c - h) - f) <= 0.13:  # cara de abajo/izquierda pegada a este muro
                return nf + ((c - h) - f) + h
            if abs(f - (c + h)) <= 0.13:  # cara de arriba/derecha pegada a este muro
                return nf - (f - (c + h)) - h
        return c * e

    nfurn = []
    for f in FURN:
        name, kind, (u, v), size = f[:4]
        if kind == 'chair':
            continue  # las sillas se rehacen alrededor de la mesa
        hu, hv = (size, size) if kind in ('cyl', 'round') else (size[0] / 2, size[1] / 2)
        if kind == 'round':
            nu, nv = u * e, v * e
        else:
            nu, nv = eje(u, hu, caras, nuevas_u), eje(v, hv, caras_v, nuevas_v)
        nfurn.append((name, kind, (round(nu, 3), round(nv, 3))) + tuple(f[3:]))
    DINNER = next(f[2] for f in nfurn if f[0] == 'dinner_table')
    dc = next(f for f in FURN if f[0] == 'silla_n')[3:6]  # lado, alto, color
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

# ------------------------------------------------------------------------------------------
# Geometría -> lista de primitivas (para SDF y para el raster de verificación)
# prim = dict(kind='box'|'cyl', c=(u,v,z), size=(sx,sy,sz) | r,h, yaw, color)
# ------------------------------------------------------------------------------------------


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
        alto = 0.45 if kind == 'chair' else h  # en la silla, bajo el asiento
        for p in faldon(kind, u, v, size, alto, f[6] if kind == 'chair' else 0.0):
            p['color'] = '0.80 0.68 0.50'  # color cartón
            P.append(p)
    if kind == 'estante':
        su, sv = size
        nu, nv = f[6]  # hacia dónde mira el lado abierto
        base = ESTANTE_NIVELES[0]
        tb = 0.02  # grosor de tablas
        # mueble cerrado de abajo (lo que ve el lidar)
        P.append(dict(kind='box', c=(u, v, base / 2), size=(su, sv, base), yaw=0.0))
        zc, hz = (base + h) / 2, h - base
        if nv:  # abierto hacia +-v: fondo a lo largo de u, costados en los extremos de u
            P.append(dict(kind='box', c=(u, v - nv * (sv / 2 - tb / 2), zc), size=(su, tb, hz),
                          yaw=0.0))
            for s_ in (-1, 1):
                P.append(dict(kind='box', c=(u + s_ * (su / 2 - tb / 2), v, zc),
                              size=(tb, sv, hz), yaw=0.0))
        else:  # abierto hacia +-u
            P.append(dict(kind='box', c=(u - nu * (su / 2 - tb / 2), v, zc), size=(tb, sv, hz),
                          yaw=0.0))
            for s_ in (-1, 1):
                P.append(dict(kind='box', c=(u, v + s_ * (sv / 2 - tb / 2), zc),
                              size=(su, tb, hz), yaw=0.0))
        # repisas (la superficie queda a la altura del nivel) y tapa de arriba
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
        s, yaw = size, f[6]  # yaw = hacia dónde mira la silla (hacia la mesa)
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
        bu, bv = rot(-s / 2 + 0.02, 0.0)  # respaldo atrás (lejos de la mesa)
        P.append(
            dict(kind='box', c=(bu, bv, seat + (h - seat) / 2), size=(0.04, s, h - seat), yaw=yaw)
        )
    for p in P:
        p.setdefault('color', color)
    return P


# ------------------------------------------------------------------------------------------
# SDF
# ------------------------------------------------------------------------------------------
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
    # Piso de madera clara de la casa (solo visual, 1 mm sobre el ground_plane).
    cx, cy = to_world(L / 2, L / 2)
    return (
        f'    <model name="piso_casa">\n      <static>true</static>\n'
        f'      <link name="l"><pose>{cx:.3f} {cy:.3f} 0.0005 0 0 0</pose>\n'
        f'        <visual name="v"><geometry><box><size>{L} {L} 0.001</size></box></geometry>'
        f'<material><ambient>0.93 0.84 0.68 1</ambient>'
        f'<diffuse>0.93 0.84 0.68 1</diffuse></material></visual>\n'
        f'      </link>\n    </model>'
    )


# ------------------------------------------------------------------------------------------
# Modelos de los objetos. Medidas APROXIMADAS (cámbialas por las de los objetos reales).
# Cada objeto es UN cuerpo libre (no static): se puede empujar y agarrar. El origen del modelo está
# en la BASE del objeto (z = 0 apoyado en la mesa); las piezas son solo visuales y la colisión es
# una forma simple (lo que importa para agarrar). Sin logos: formas y colores.
# pieza = (forma, (x, y, z_centro), medidas, color, transparencia)
#   forma 'cyl': medidas = (radio, alto); 'box': medidas = (sx, sy, sz)
# ------------------------------------------------------------------------------------------
NARANJA, ROJO, ROJO_T = '0.86 0.40 0.22', '0.80 0.10 0.12', '0.82 0.08 0.08'
MODELOS = {
    # Botella PET de ~500 ml: 6.6 cm de diámetro, 22 cm de alto, 0.55 kg llena.
    'botella': dict(
        masa=0.55,
        col=('cyl', (0.033, 0.22)),
        piezas=[
            ('cyl', (0, 0, 0.075), (0.033, 0.150), NARANJA, 0),  # cuerpo
            ('cyl', (0, 0, 0.160), (0.027, 0.020), NARANJA, 0),  # hombro
            ('cyl', (0, 0, 0.182), (0.015, 0.025), '0.75 0.78 0.80', 0.4),  # cuello (plástico)
            ('cyl', (0, 0, 0.192), (0.019, 0.004), '0.75 0.78 0.80', 0.2),  # anillo del cuello
            ('cyl', (0, 0, 0.2075), (0.016, 0.025), ROJO, 0),  # tapa
            ('cyl', (0, 0, 0.130), (0.034, 0.012), ROJO, 0),  # franjas de la etiqueta
            ('cyl', (0, 0, 0.0625), (0.034, 0.015), '0.85 0.75 0.20', 0),  # amarilla
            ('cyl', (0, 0, 0.0500), (0.034, 0.010), '0.30 0.60 0.25', 0),  # verde
            ('cyl', (0, 0, 0.0375), (0.034, 0.015), ROJO, 0),
        ],
    ),
    # Taza: 8.6 cm de diámetro, 9.5 cm de alto, asa de ~3 cm hacia +x, 0.30 kg.
    'taza': dict(
        masa=0.30,
        col=('cyl', (0.043, 0.095)),
        piezas=[
            ('cyl', (0, 0, 0.0475), (0.043, 0.095), ROJO_T, 0),  # cuerpo
            ('cyl', (0, 0, 0.0945), (0.037, 0.002), '0.95 0.95 0.95', 0),  # interior blanco
            ('box', (0.073, 0, 0.048), (0.012, 0.014, 0.066), ROJO_T, 0),  # asa: vertical
            ('box', (0.058, 0, 0.075), (0.032, 0.014, 0.012), ROJO_T, 0),  # asa: arriba
            ('box', (0.058, 0, 0.021), (0.032, 0.014, 0.012), ROJO_T, 0),  # asa: abajo
        ],
    ),
    # Paquete de galletas acostado: 19 x 6.2 x 3.5 cm, 0.13 kg. Bajito: la garra tiene que bajar
    # más para agarrarlo (y tapa más el lidar).
    'galletas': dict(
        masa=0.13,
        col=('box', (0.19, 0.062, 0.035)),
        piezas=[
            ('box', (0, 0, 0.0175), (0.19, 0.062, 0.035), '0.70 0.08 0.10', 0),  # envoltura
            ('box', (0.02, 0, 0.0355), (0.12, 0.030, 0.001), '0.85 0.65 0.30', 0),  # franja
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
    zc = med[1] / 2 if forma == 'cyl' else med[2] / 2  # centro de la colisión (origen = base)
    ixx, iyy, izz = inercia(forma, m['masa'], med)
    vis = []
    for k, (f, (px, py, pz), md, col, tr) in enumerate(m['piezas']):
        vis.append(
            f'        <visual name="v{k}"><pose>{px} {py} {pz} 0 0 0</pose>'
            f'<geometry>{geom_obj(f, md)}</geometry><transparency>{tr}</transparency>'
            f'<material><ambient>{col} 1</ambient><diffuse>{col} 1</diffuse></material></visual>'
        )
    # fricción alta (mu = 1) para que no resbale en la garra ni en la mesa
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
        return med[0] + (0.036 if o['modelo'] == 'taza' else 0.0)  # la taza: + el asa
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
        nu, nv = lado[mueble][2]  # dirección mueble -> Milo
        if kind in ('round', 'cyl'):
            half = size
        else:
            half = abs(nu) * size[0] / 2 + abs(nv) * size[1] / 2
        # punto de la cara frente a Milo, después hacia adentro (prof; negativo = afuera, en el
        # suelo) y a lo largo de la cara (lat)
        tu, tv = nv, -nu  # a la izquierda de Milo cuando mira el mueble (mira hacia -n)
        u = cu + nu * (half - prof) + tu * lat
        v = cv + nv * (half - prof) + tv * lat
        yaw_w = math.atan2(-nv, -nu) + yaw
        yaw_w = math.atan2(math.sin(yaw_w), math.cos(yaw_w))
        z = nivel if nivel is not None else h
        o = dict(nombre=nombre, modelo=modelo, mueble=mueble, donde=donde, prof=prof, lateral=lat,
                 u=u, v=v, yaw=yaw_w, z=round(z + 0.001, 3))
        o['x'], o['y'] = to_world(u, v)
        # pre-agarre (para la validación con el PPO): sobre la normal de la cara por donde llega
        # Milo, a lo largo de la cara donde está el objeto, con la punta del robot a DIST_PRE del
        # borde del mueble; mirando hacia el mueble. Solo para objetos sobre muebles.
        if donde == 'mueble':
            d = half + DIST_PRE + PUNTA
            o['pre'] = (*to_world(cu + nu * d + tu * lat, cv + nv * d + tv * lat),
                        math.atan2(-nv, -nu))
        # profundidad efectiva y si el objeto cabe entero en su superficie
        if donde == 'suelo':
            o['prof_ef'], o['cabe'] = None, True
        elif kind == 'round':
            rho = math.hypot(u - cu, v - cv)
            o['prof_ef'] = size - rho
            o['cabe'] = rho + medio_ancho(o, math.atan2(v - cv, u - cu)) <= size
        elif kind == 'estante':
            o['prof_ef'] = prof
            fondo = 2 * half - 0.02  # hasta la tabla del fondo
            ancho = abs(nv) * size[0] + abs(nu) * size[1] - 0.04  # entre los costados
            ang_in = math.atan2(-nv, -nu)
            o['cabe'] = (prof + medio_ancho(o, ang_in) <= fondo
                         and abs(lat) + medio_ancho(o, ang_in + math.pi / 2) <= ancho / 2)
        else:  # mesa rectangular: el borde más cercano de los 4
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


# ------------------------------------------------------------------------------------------
# Raster (verificación + PNG). Coordenadas casa, margen de 1.5 m afuera.
# ------------------------------------------------------------------------------------------
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
            for j in range(1, NU):  # barrido izq->der
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
    allp += [objeto_prim(o) for o in objs]  # los del suelo cuentan como obstáculo físico
    write_world(walls, furn, objs)
    write_objetos_yaml(objs)

    # --- Poses objetivo
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
        yaw = math.atan2(-nv, -nu)  # mirando al mueble
        goals.append(dict(name=fname, id=rid, room=room, place=place, u=gu, v=gv, yaw=yaw))
    for rname, (ru, rv, ryaw) in ROOMS.items():
        goals.append(dict(name=rname, id=None, room=rname, place=False, u=ru, v=rv, yaw=ryaw))

    # --- Verificación
    occ_phys = raster(allp)  # choque físico (todo hasta 1.3 m)
    occ_lidar = raster(allp, LIDAR_Z)  # lo que ve el lidar
    d_phys = dist_field(occ_phys)
    ok = True
    # (a) cada pose: footprint libre (+3 cm) y la punta lejos del stop del filtro de seguridad
    for g in goals:
        fp = footprint_cells(g['u'], g['v'], g['yaw'])
        clear = float(d_phys[fp].min())
        g['clear'] = clear
        if clear < 0.03:
            print(f'  [MAL] {g["name"]}: el footprint choca (holgura {clear:.2f} m)')
            ok = False
        # zona de parada del filtro de seguridad (caja del robot + 0.30 m adelante + 0.10 a los
        # lados), contra lo que ve el lidar: si hay algo ahí, el filtro no dejaría llegar.
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
    # (b) alcanzabilidad: BFS desde la partida con un disco de 0.27 m (Milo pasando de frente)
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
    # puertas: holgura libre
    for dn, (du_, dv_) in DOORS.items():
        print(f'  puerta {dn:24s} libre a cada lado del centro: {d_phys[idx(du_, dv_)]:.2f} m')

    # (c) objetos: enteros en su superficie y en la franja "solo relajando" del filtro
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

    # (d) tus lugares anotados (casa_lugares.yaml): ¿siguen libres con las mesas nuevas?
    try:
        import yaml

        with open(LUGARES) as fh:
            lug = yaml.safe_load(fh) or {}
        lug = lug.get('lugares', lug)  # formato plano o completo
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

    # --- PNG
    try:
        from PIL import Image, ImageDraw

        sc = 4  # 1 celda = 4 px -> 2 cm = 4 px
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

        # número del rulebook sobre cada mueble (como en el plano)
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
