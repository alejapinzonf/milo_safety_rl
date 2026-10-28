#!/usr/bin/env python3
"""Evalúa métodos (baselines y modelos entrenados) en los 3 escenarios, con las MISMAS escenas.

Uso (desde rl/):
  python3 scripts/evaluar.py                       # baselines + todo lo que haya en runs/
  python3 scripts/evaluar.py --episodios 300 --solo nominal minimo reglas generalista
  python3 scripts/evaluar.py --modelo mio=runs/generalista/semilla_0/final.zip

Cada episodio usa la semilla 100000 + i para todos los métodos (comparación pareada). Con los
modelos, cada semilla de entrenamiento se evalúa por separado (columna semilla_modelo) para poder
poner barras de error entre semillas.

Salida: resultados/episodios.csv (una fila por episodio) y resultados/resumen.csv.
"""
import argparse
import csv
import glob
import os
import sys
from multiprocessing import Pool

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))

import numpy as np  # noqa: E402

from milo_rl import config as C  # noqa: E402
from milo_rl.baselines import BASELINES, EXTRA  # noqa: E402
from milo_rl.entorno import MiloEnv  # noqa: E402

CAMPOS = ['exito', 'agarro', 'motivo', 't', 'choque_mueble', 'contacto_persona',
          'violaciones_iso', 'intrusiones_iso', 'd_min_persona', 't_intimo', 't_personal',
          'exposicion', 'costo_soc', 't_ciego_mov', 'intervenciones', 'giros_bloqueados',
          'despejes', 'recuperaciones', 'recorrido', 'tipo_personas', 'donde', 'facil', 'prof_ef',
          'oculta', 'choque_oculta', 'vio_oculta']


def cfg_para(nombre_run, escenario):
    """Las ablaciones se evalúan con su misma configuración del entorno."""
    cfg = C.ConfigEntorno(escenario=C.TODOS[escenario])
    if 'sin_escudo' in nombre_run:
        cfg.escudo.activo = False
    if 'sin_memoria' in nombre_run:
        cfg.escudo.usar_memoria = False
    if 'sin_personas_obs' in nombre_run:
        cfg.usar_personas_en_obs = False
    if 'un_sector' in nombre_run:
        cfg.un_sector = True
    return cfg


def correr(tarea):
    metodo, semilla_modelo, ruta, escenario, i0, n = tarea
    env = MiloEnv(cfg_para(metodo, escenario), semilla=0)
    if ruta:
        from stable_baselines3 import PPO
        modelo = PPO.load(ruta, device='cpu')

        def pol(env, obs):
            return modelo.predict(obs, deterministic=True)[0]
    else:
        base = BASELINES[metodo]

        def pol(env, obs):
            return base(env)
    filas = []
    for i in range(i0, i0 + n):
        obs, _ = env.reset(seed=100000 + i)
        while True:
            obs, r, te, tr, info = env.step(pol(env, obs))
            if te or tr:
                break
        m = info['episodio']
        filas.append([metodo, semilla_modelo, escenario, i] + [m[k] for k in CAMPOS])
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--episodios', type=int, default=200, help='por escenario y método')
    ap.add_argument('--runs', default=os.path.join(os.path.dirname(AQUI), 'runs'))
    ap.add_argument('--solo', nargs='*', help='nombres de métodos a evaluar')
    ap.add_argument('--modelo', action='append', default=[], help='nombre=ruta.zip')
    ap.add_argument('--escenarios', nargs='*', default=list(C.ESCENARIOS))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(AQUI), 'resultados'))
    ap.add_argument('--procesos', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    metodos = [(b, -1, None) for b in BASELINES if b not in EXTRA or (args.solo and b in args.solo)]
    for ruta in sorted(glob.glob(os.path.join(args.runs, '*', 'semilla_*', 'final.zip'))):
        run = ruta.split(os.sep)[-3]
        s = int(ruta.split(os.sep)[-2].split('_')[1])
        metodos.append((run, s, ruta))
    for txt in args.modelo:
        n, r = txt.split('=', 1)
        metodos.append((n, 0, r))
    if args.solo:
        metodos = [m for m in metodos if m[0] in args.solo]
    if not metodos:
        sys.exit('No hay métodos para evaluar.')
    print('Métodos:', sorted({(m[0]) for m in metodos}))

    trozo = 25
    tareas = [(m, s, r, e, i, min(trozo, args.episodios - i)) for m, s, r in metodos
              for e in args.escenarios for i in range(0, args.episodios, trozo)]
    os.makedirs(args.out, exist_ok=True)
    filas = []
    with Pool(args.procesos) as pool:
        for k, res in enumerate(pool.imap_unordered(correr, tareas)):
            filas += res
            print(f'\r{k + 1}/{len(tareas)} lotes', end='', flush=True)
    print()
    ruta = os.path.join(args.out, 'episodios.csv')
    with open(ruta, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['metodo', 'semilla_modelo', 'escenario', 'episodio'] + CAMPOS)
        w.writerows(sorted(filas, key=lambda f: (f[0], f[1], f[2], f[3])))
    print('->', ruta)
    resumir(ruta, os.path.join(args.out, 'resumen.csv'))


def resumir(ruta_ep, ruta_out):
    import pandas as pd
    df = pd.read_csv(ruta_ep)
    df['choque'] = (df['motivo'] == 'choque').astype(float)
    g = df.groupby(['metodo', 'escenario'])
    res = g.agg(episodios=('exito', 'size'), exito=('exito', 'mean'), agarro=('agarro', 'mean'),
                choque=('choque', 'mean'), viol_iso=('violaciones_iso', 'mean'),
                exposicion=('exposicion', 'mean'), costo_soc=('costo_soc', 'mean'),
                t=('t', 'mean'), t_ciego_mov=('t_ciego_mov', 'mean'),
                despejes=('despejes', 'mean')).reset_index()
    res.to_csv(ruta_out, index=False, float_format='%.3f')
    with np.printoptions(precision=3):
        print(res.to_string(index=False))
    print('->', ruta_out)


if __name__ == '__main__':
    main()
