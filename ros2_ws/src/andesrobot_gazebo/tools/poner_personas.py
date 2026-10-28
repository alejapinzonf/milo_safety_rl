#!/usr/bin/env python3
"""Pone personas QUIETAS en la casa para la validación con personas (E5): genera
worlds/robocup_casa_personas.world (= robocup_casa.world + las personas) y
worlds/robocup_casa_personas.yaml (dónde quedó cada una; lo lee el nodo del PPO).

  python3 tools/poner_personas.py

Cada persona se para al lado del pre-agarre de un objeto (ver robocup_casa_objetos.yaml, campo
`pre`), mirando hacia donde llega Milo, en un lugar libre (revisado contra los muebles del
.world). Los modelos salen de human-gazebo vía humanos.py (una persona quieta con colisión, que el
lidar sí ve: a 0.36 m de alto corta las dos piernas).
"""
import math
import os
import re
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import humanos  # noqa: E402

MUNDO = os.path.join(HERE, '..', 'worlds', 'robocup_casa.world')
OBJETOS = os.path.join(HERE, '..', 'worlds', 'robocup_casa_objetos.yaml')
SALIDA = os.path.join(HERE, '..', 'worlds', 'robocup_casa_personas')
# nombre de la persona: (objeto junto a cuyo pre-agarre se para, sujeto de human-gazebo, pose)
PERSONAS = {
    'persona_comedor': ('taza_comedor', 1, 'de_pie'),
    'persona_lavanderia': ('botella_laundry', 3, 'de_pie'),
    'persona_librero': ('botella_librero', 5, 'senalando'),
}
RES = 0.05
LIBRE = 0.40  # [m] mínima distancia de la persona (su centro) a cualquier mueble


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


def libre(xs, dist, p):
    i, j = int(round((p[1] - xs[0]) / RES)), int(round((p[0] - xs[0]) / RES))
    return 0 <= i < len(xs) and 0 <= j < len(xs) and dist[i, j] >= LIBRE


def main():
    from scipy.ndimage import distance_transform_edt
    sdf = open(MUNDO).read()
    xs, occ = ocupacion(sdf)
    dist = distance_transform_edt(~occ) * RES
    objs = yaml.safe_load(open(OBJETOS))
    modelos, salida = [], {}
    for nombre, (obj, sujeto, pose) in PERSONAS.items():
        px, py, pyaw = objs[obj]['pre']
        elegido = None
        for lat in (1.1, 1.3, 1.5, 0.9):
            for lado in (1, -1):
                for fw in (0.3, 0.0, 0.6, -0.3):
                    c = np.array([px, py]) + fw * np.array([math.cos(pyaw), math.sin(pyaw)]) \
                        + lado * lat * np.array([-math.sin(pyaw), math.cos(pyaw)])
                    if elegido is None and libre(xs, dist, c):
                        elegido = c
        if elegido is None:
            sys.exit(f'{nombre}: no hay lugar libre junto al pre-agarre de {obj}')
        yaw = math.atan2(py - elegido[1], px - elegido[0])  # mira hacia donde llega Milo
        piezas = humanos.piezas_en_pose(sujeto, pose, float(elegido[0]), float(elegido[1]), yaw)
        modelos.append(humanos.modelo_sdf(nombre, piezas))
        salida[nombre] = dict(x=round(float(elegido[0]), 3), y=round(float(elegido[1]), 3),
                              yaw=round(yaw, 3), sujeto=sujeto, pose=pose, junto_a=obj)
        print(f'{nombre}: ({elegido[0]:.2f}, {elegido[1]:.2f}) junto al pre-agarre de {obj}')
    nuevo = sdf.replace('  </world>', '\n'.join(modelos) + '\n  </world>')
    with open(SALIDA + '.world', 'w') as fh:
        fh.write(nuevo)
    with open(SALIDA + '.yaml', 'w') as fh:
        fh.write('# GENERADO por tools/poner_personas.py — NO editar a mano.\n'
                 '# Personas quietas de robocup_casa_personas.world (x, y en el mundo = frame map).\n')
        yaml.safe_dump(salida, fh, sort_keys=False)
    print('->', os.path.relpath(SALIDA + '.world'), 'y', os.path.relpath(SALIDA + '.yaml'))


if __name__ == '__main__':
    main()
