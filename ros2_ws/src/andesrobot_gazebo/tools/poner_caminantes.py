#!/usr/bin/env python3
"""Pone personas que CAMINAN en la casa (validación E5 con personas en movimiento).

  python3 tools/poner_caminantes.py

Genera worlds/robocup_casa_caminando.world (= robocup_casa.world + un caminante por zona + el
plugin gazebo_ros_state, que da el servicio para moverlos) y worlds/robocup_casa_caminando.yaml
(la ruta de cada uno; la lee rl/ros/nodo_rl.py, que los mueve y le pasa al filtro sus posiciones).

Cada caminante va y viene por un tramo recto que CRUZA el camino hacia el pre-agarre de un objeto
(a 0.9-2.2 m de él), por un lugar libre de muebles (>= 0.45 m a lo largo de toda la ruta).
El modelo es un cilindro con colisión (lo ve el lidar) + figura humana simple de solo visual.
Es cinemático: no lo empuja nadie, el nodo le fija la pose a 20 Hz.
"""
import math
import os
import re
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MUNDO = os.path.join(HERE, '..', 'worlds', 'robocup_casa.world')
OBJETOS = os.path.join(HERE, '..', 'worlds', 'robocup_casa_objetos.yaml')
SALIDA = os.path.join(HERE, '..', 'worlds', 'robocup_casa_caminando')
CAMINANTES = {
    'caminante_comedor': 'taza_comedor',
    'caminante_lavanderia': 'botella_laundry',
    'caminante_librero': 'botella_librero',
}
VEL = 0.8
PAUSA = 1.0
RES = 0.05
LIBRE = 0.45
PLUGIN_ESTADO = """    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros>
      <update_rate>10.0</update_rate>
    </plugin>
"""


def ocupacion(sdf):
    """Ocupación 2D (0 a 1.3 m de alto) de todos los links con colisión del .world."""
    prims = []
    for m in re.finditer(r'<link name="[^"]*"><pose>([^<]*)</pose>\s*<collision name="[^"]*">'
                         r'<geometry>(.*?)</geometry>', sdf):
        x, y, z, _, _, yaw = map(float, m.group(1).split())
        g = m.group(2)
        if '<box>' in g:
            a, b, c = map(float, re.search(r'<size>([^<]*)</size>', g).group(1).split())
            prims.append(('box', x, y, z, yaw, (a, b), c))
        elif '<cylinder>' in g:
            r = float(re.search(r'<radius>([^<]*)</radius>', g).group(1))
            h = float(re.search(r'<length>([^<]*)</length>', g).group(1))
            prims.append(('cyl', x, y, z, yaw, r, h))
    xs = np.arange(-9, 9, RES)
    X, Y = np.meshgrid(xs, xs)
    occ = np.zeros(X.shape, bool)
    for kind, x, y, z, yaw, d, h in prims:
        if z - h / 2 > 1.3:
            continue
        dx, dy = X - x, Y - y
        if kind == 'cyl':
            occ |= dx * dx + dy * dy <= d * d
        else:
            c, s = math.cos(-yaw), math.sin(-yaw)
            occ |= (np.abs(dx * c - dy * s) <= d[0] / 2) & (np.abs(dx * s + dy * c) <= d[1] / 2)
    return xs, occ


def libre_a_lo_largo(xs, dist, a, b):
    for s in np.linspace(0, 1, int(np.hypot(*(b - a)) / RES) + 2):
        p = a + s * (b - a)
        i, j = int(round((p[1] - xs[0]) / RES)), int(round((p[0] - xs[0]) / RES))
        if not (0 <= i < len(xs) and 0 <= j < len(xs)) or dist[i, j] < LIBRE:
            return False
    return True


def dist_punto_segmento(p, a, b):
    ab = b - a
    s = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.hypot(*(p - (a + s * ab))))


def elegir_ruta(xs, dist, pre):
    px, py, pyaw = pre
    p = np.array([px, py])
    f = np.array([math.cos(pyaw), math.sin(pyaw)])
    n = np.array([-f[1], f[0]])
    for largo in (3.0, 2.4, 1.8):
        for avance in (-1.4, -1.8, -1.0, -2.2, -0.6):
            for desfase in (0.0, 0.6, -0.6, 1.2, -1.2):
                for orient in (n, f):
                    c = p + avance * f + desfase * n
                    a, b = c - orient * largo / 2, c + orient * largo / 2
                    if not libre_a_lo_largo(xs, dist, a, b):
                        continue
                    d = dist_punto_segmento(p, a, b)
                    if 0.9 <= d <= 2.2 and min(np.hypot(*(a - p)), np.hypot(*(b - p))) >= 1.0:
                        return a, b
    return None


def modelo_sdf(nombre, a, b):
    yaw = math.atan2(b[1] - a[1], b[0] - a[0])
    mat = lambda r, g, bl: (f'<material><ambient>{r} {g} {bl} 1</ambient>'
                            f'<diffuse>{r} {g} {bl} 1</diffuse></material>')
    vis = [
        ('piernas_i', '<cylinder><radius>0.07</radius><length>0.80</length></cylinder>', 0.0, 0.10, 0.40, mat(0.18, 0.22, 0.35)),
        ('piernas_d', '<cylinder><radius>0.07</radius><length>0.80</length></cylinder>', 0.0, -0.10, 0.40, mat(0.18, 0.22, 0.35)),
        ('torso', '<box><size>0.22 0.40 0.60</size></box>', 0.0, 0.0, 1.10, mat(0.75, 0.25, 0.20)),
        ('cabeza', '<sphere><radius>0.11</radius></sphere>', 0.0, 0.0, 1.55, mat(0.85, 0.70, 0.58)),
        ('brazo_i', '<cylinder><radius>0.04</radius><length>0.60</length></cylinder>', 0.0, 0.24, 1.08, mat(0.75, 0.25, 0.20)),
        ('brazo_d', '<cylinder><radius>0.04</radius><length>0.60</length></cylinder>', 0.0, -0.24, 1.08, mat(0.75, 0.25, 0.20)),
    ]
    v = ''.join(f'<visual name="{n}"><pose>{x} {y} {z} 0 0 0</pose><geometry>{g}</geometry>{m}</visual>'
                for n, g, x, y, z, m in vis)
    return f"""    <model name="{nombre}">
      <static>false</static>
      <pose>{a[0]:.3f} {a[1]:.3f} 0 0 0 {yaw:.4f}</pose>
      <link name="cuerpo">
        <gravity>false</gravity>
        <kinematic>true</kinematic>
        <self_collide>false</self_collide>
        <inertial><mass>70</mass><inertia><ixx>10</ixx><iyy>10</iyy><izz>2</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><pose>0 0 0.85 0 0 0</pose><geometry><cylinder><radius>0.20</radius><length>1.70</length></cylinder></geometry></collision>
        {v}
      </link>
    </model>"""


def main():
    from scipy.ndimage import distance_transform_edt
    sdf = open(MUNDO).read()
    xs, occ = ocupacion(sdf)
    dist = distance_transform_edt(~occ) * RES
    objs = yaml.safe_load(open(OBJETOS))
    modelos, salida = [], {}
    for nombre, obj in CAMINANTES.items():
        r = elegir_ruta(xs, dist, objs[obj]['pre'])
        if r is None:
            sys.exit(f'{nombre}: no hay ruta libre cruzando el pre-agarre de {obj}')
        a, b = r
        modelos.append(modelo_sdf(nombre, a, b))
        salida[nombre] = dict(tipo='caminante', ruta=[[round(float(a[0]), 3), round(float(a[1]), 3)],
                                                      [round(float(b[0]), 3), round(float(b[1]), 3)]],
                              vel=VEL, pausa=PAUSA, cruza=obj)
        print(f'{nombre}: ({a[0]:.2f},{a[1]:.2f}) <-> ({b[0]:.2f},{b[1]:.2f}) cruza el pre-agarre de {obj}')
    nuevo = sdf.replace('  </world>', PLUGIN_ESTADO + '\n'.join(modelos) + '\n  </world>')
    with open(SALIDA + '.world', 'w') as fh:
        fh.write(nuevo)
    with open(SALIDA + '.yaml', 'w') as fh:
        fh.write('# GENERADO por tools/poner_caminantes.py — NO editar a mano.\n'
                 '# Personas que caminan de ida y vuelta por `ruta` (x, y en el mundo = frame map).\n')
        yaml.safe_dump(salida, fh, sort_keys=False)
    print('->', os.path.relpath(SALIDA + '.world'), 'y', os.path.relpath(SALIDA + '.yaml'))


if __name__ == '__main__':
    main()
