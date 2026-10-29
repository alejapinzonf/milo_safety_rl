#!/usr/bin/env python3
"""Resume los CSV de la validación en Gazebo (ros2_ws/resultados/rl_<modo>_<mundo>_s<semilla>_*.csv).

  python3 rl/scripts/gazebo_tabla.py ros2_ws/resultados/rl_*.csv
Una fila por método (el PPO promediado entre semillas, con el rango entre semillas).
Solo usa la librería estándar (no hace falta instalar pandas).
"""
import csv
import math
import sys
from collections import defaultdict
from statistics import mean


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


corridas = defaultdict(list)
for ruta in sys.argv[1:]:
    semilla = int(ruta.split('_s')[-1].split('_')[0])
    mundo = ruta.split('rl_')[-1].split('_s')[0].split('_', 1)[1]
    with open(ruta, newline='') as fh:
        for fila in csv.DictReader(fh):
            corridas[(mundo, fila['modo'], semilla)].append(fila)

por_corrida = []
for (mundo, modo, semilla), filas in sorted(corridas.items()):
    col = lambda k: [num(f[k]) for f in filas]  # noqa: E731
    por_corrida.append(dict(
        mundo=mundo, modo=modo, semilla=semilla, n=len(filas),
        exito=mean(f['resultado'] == 'exito' for f in filas), t=mean(col('t_total')),
        exposicion=mean(col('exposicion')), costo_soc=mean(col('costo_soc')),
        d_min_mueble=min(col('d_min_mueble')), d_min_persona=min(col('d_min_persona')),
        choques=sum(col('choque')), viol_iso=sum(col('violaciones_iso'))))

grupos = defaultdict(list)
for r in por_corrida:
    grupos[(r['mundo'], r['modo'])].append(r)
CAMPOS = ['mundo', 'modo', 'corridas', 'n', 'exito', 'exito_min', 'exito_max', 't', 'exposicion',
          'costo_soc', 'd_min_mueble', 'd_min_persona', 'choques', 'viol_iso']
res = []
for (mundo, modo), rs in sorted(grupos.items()):
    res.append(dict(
        mundo=mundo, modo=modo, corridas=len(rs), n=sum(r['n'] for r in rs),
        exito=mean(r['exito'] for r in rs), exito_min=min(r['exito'] for r in rs),
        exito_max=max(r['exito'] for r in rs), t=mean(r['t'] for r in rs),
        exposicion=mean(r['exposicion'] for r in rs), costo_soc=mean(r['costo_soc'] for r in rs),
        d_min_mueble=min(r['d_min_mueble'] for r in rs),
        d_min_persona=min(r['d_min_persona'] for r in rs),
        choques=sum(r['choques'] for r in rs), viol_iso=sum(r['viol_iso'] for r in rs)))


def fmt(v):
    return f'{v:.3f}'.rstrip('0').rstrip('.') if isinstance(v, float) and math.isfinite(v) \
        else str(v)


filas = [[fmt(r[c]) for c in CAMPOS] for r in res]
anchos = [max(len(c), *(len(f[i]) for f in filas)) for i, c in enumerate(CAMPOS)]
print('  '.join(c.rjust(a) for c, a in zip(CAMPOS, anchos)))
for f in filas:
    print('  '.join(x.rjust(a) for x, a in zip(f, anchos)))
with open('resumen_gazebo.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(CAMPOS)
    w.writerows(filas)
print('-> resumen_gazebo.csv')
