#!/usr/bin/env python3
"""Hace un GIF de un episodio (para revisar qué hace la política o para presentaciones).

Uso (desde rl/):
  python3 scripts/ver_episodio.py --metodo reglas --escenario caminando --episodio 3
  python3 scripts/ver_episodio.py --modelo runs/generalista/semilla_0/final.zip --escenario quietas
El episodio i usa la misma escena que en evaluar.py (semilla 100000 + i).
Salida: resultados/gifs/<metodo>_<escenario>_<i>.gif
"""
import argparse
import os
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from milo_rl import config as C  # noqa: E402
from milo_rl.baselines import BASELINES  # noqa: E402
from milo_rl.entorno import MiloEnv  # noqa: E402
from milo_rl.viz import dibujar, gif  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--metodo', default='reglas', choices=list(BASELINES))
    ap.add_argument('--modelo', help='ruta a un .zip entrenado (en vez de --metodo)')
    ap.add_argument('--escenario', default='sin_personas', choices=list(C.TODOS))
    ap.add_argument('--episodio', type=int, default=0)
    ap.add_argument('--cada', type=int, default=2, help='un cuadro cada N pasos (0.2 s)')
    args = ap.parse_args()
    cfg = C.ConfigEntorno(escenario=C.TODOS[args.escenario])
    if 'sin_memoria' in (args.metodo if not args.modelo else args.modelo):
        cfg.escudo.usar_memoria = False
    env = MiloEnv(cfg, semilla=0)
    if args.modelo:
        from stable_baselines3 import PPO
        m = PPO.load(args.modelo, device='cpu')
        nombre = os.path.basename(os.path.dirname(os.path.dirname(args.modelo)))

        def pol(obs):
            return m.predict(obs, deterministic=True)[0]
    else:
        nombre = args.metodo

        def pol(obs):
            return BASELINES[args.metodo](env)
    obs, _ = env.reset(seed=100000 + args.episodio)
    tmp = tempfile.mkdtemp()
    frames, k = [], 0
    while True:
        if k % args.cada == 0:
            fig, ax = plt.subplots(figsize=(5.5, 4.6))
            dibujar(env, ax)
            f = os.path.join(tmp, f'{k:05d}.png')
            fig.savefig(f, dpi=70)
            plt.close(fig)
            frames.append(f)
        obs, r, te, tr, info = env.step(pol(obs))
        k += 1
        if te or tr:
            break
    out = os.path.join(os.path.dirname(AQUI), 'resultados', 'gifs')
    os.makedirs(out, exist_ok=True)
    ruta = os.path.join(out, f'{nombre}_{args.escenario}_{args.episodio}.gif')
    gif(frames, ruta, ms=int(200 * args.cada / 2))
    print(info['episodio']['motivo'], '->', ruta)


if __name__ == '__main__':
    main()
