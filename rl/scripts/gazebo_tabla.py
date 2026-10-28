#!/usr/bin/env python3
"""Resume los CSV de la validación en Gazebo (ros2_ws/resultados/rl_<modo>_<mundo>_s<semilla>_*.csv).

  python3 rl/scripts/gazebo_tabla.py ros2_ws/resultados/rl_*.csv
Una fila por método (el PPO promediado entre semillas, con el rango entre semillas).
"""
import sys

import pandas as pd

dfs = []
for ruta in sys.argv[1:]:
    d = pd.read_csv(ruta)
    d['semilla'] = int(ruta.split('_s')[-1].split('_')[0])
    d['mundo'] = ruta.split('rl_')[-1].split('_s')[0].split('_', 1)[1]
    dfs.append(d)
d = pd.concat(dfs, ignore_index=True)
d['exito'] = (d.resultado == 'exito').astype(float)
por_corrida = d.groupby(['mundo', 'modo', 'semilla']).agg(
    n=('exito', 'size'), exito=('exito', 'mean'), t=('t_total', 'mean'),
    exposicion=('exposicion', 'mean'), costo_soc=('costo_soc', 'mean'),
    d_min_mueble=('d_min_mueble', 'min'), d_min_persona=('d_min_persona', 'min'),
    choques=('choque', 'sum'), viol_iso=('violaciones_iso', 'sum')).reset_index()
res = por_corrida.groupby(['mundo', 'modo']).agg(
    corridas=('semilla', 'size'), n=('n', 'sum'), exito=('exito', 'mean'),
    exito_min=('exito', 'min'), exito_max=('exito', 'max'), t=('t', 'mean'),
    exposicion=('exposicion', 'mean'), costo_soc=('costo_soc', 'mean'),
    d_min_mueble=('d_min_mueble', 'min'), d_min_persona=('d_min_persona', 'min'),
    choques=('choques', 'sum'), viol_iso=('viol_iso', 'sum')).reset_index()
print(res.round(3).to_string(index=False))
res.round(3).to_csv('resumen_gazebo.csv', index=False)
print('-> resumen_gazebo.csv')
