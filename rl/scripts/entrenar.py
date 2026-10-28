#!/usr/bin/env python3
"""Entrena el PPO (Lagrangiano + escudo) del paper.

Uso (desde la carpeta rl/):
  python3 scripts/entrenar.py --semilla 0                  # generalista: las 3 etapas seguidas
  python3 scripts/entrenar.py --semilla 0 --etapa 2        # solo la etapa 2 (carga la 1)
  python3 scripts/entrenar.py --semilla 0 --especialista caminando   # especialista (desde cero)
  python3 scripts/entrenar.py --semilla 0 --mezcla-desde-cero        # ablación: sin currículo
  python3 scripts/entrenar.py --semilla 0 --sin-escudo               # ablación (también:
      --sin-lagrangiano, --sin-memoria, --sin-personas-obs, --un-sector)
  python3 scripts/entrenar.py --prueba                     # 20 mil pasos, para ver que todo anda

Cada corrida queda en runs/<nombre>/semilla_<s>/: modelo de cada etapa (.zip), lambdas (.json),
log por rollout (progreso.csv) y la configuración (config.json). TensorBoard en runs/tb/.
"""
import argparse
import dataclasses
import json
import os
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor  # noqa: E402

from milo_rl import config as C  # noqa: E402
from milo_rl.entorno import MiloEnv  # noqa: E402
from milo_rl.lagrangiano import CallbackLagrangiano  # noqa: E402

PPO_KW = dict(n_steps=1024, batch_size=512, n_epochs=10, learning_rate=3e-4, gamma=0.99,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.002, vf_coef=0.5, max_grad_norm=0.5,
              policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]),
                                 activation_fn=torch.nn.Tanh, log_std_init=-0.5))


def _hay_tb():
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


def hacer_cfg(args, escenario):
    cfg = C.ConfigEntorno(escenario=escenario)
    cfg.escudo = dataclasses.replace(cfg.escudo, activo=not args.sin_escudo,
                                     usar_memoria=not args.sin_memoria)
    cfg.usar_personas_en_obs = not args.sin_personas_obs
    cfg.un_sector = args.un_sector
    return cfg


def hacer_envs(cfg, n, semilla):
    def f(i):
        def _f():
            return MiloEnv(cfg, semilla=semilla * 1000 + i)
        return _f
    Vec = SubprocVecEnv if n > 1 else DummyVecEnv
    return VecMonitor(Vec([f(i) for i in range(n)]))


def nombre_corrida(args):
    if args.especialista:
        base = f'especialista_{args.especialista}'
    elif args.mezcla_desde_cero:
        base = 'mezcla_desde_cero'
    else:
        base = 'generalista'
    for flag in ('sin_escudo', 'sin_lagrangiano', 'sin_memoria', 'sin_personas_obs', 'un_sector'):
        if getattr(args, flag):
            base += '_' + flag
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--semilla', type=int, default=0)
    ap.add_argument('--etapa', type=int, default=0, help='1..3; 0 = todas desde la 1')
    ap.add_argument('--especialista', choices=list(C.ESCENARIOS))
    ap.add_argument('--mezcla-desde-cero', action='store_true')
    ap.add_argument('--sin-escudo', action='store_true')
    ap.add_argument('--sin-lagrangiano', action='store_true',
                    help='lambdas fijos (los iniciales), sin ascenso dual')
    ap.add_argument('--sin-memoria', action='store_true')
    ap.add_argument('--sin-personas-obs', action='store_true')
    ap.add_argument('--un-sector', action='store_true')
    ap.add_argument('--n-envs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--escala-pasos', type=float, default=1.0, help='multiplica los pasos')
    ap.add_argument('--dir', default=os.path.join(os.path.dirname(AQUI), 'runs'))
    ap.add_argument('--prueba', action='store_true')
    args = ap.parse_args()
    if args.prueba:
        args.escala_pasos = 20_000 / 2_750_000
        args.n_envs = min(args.n_envs, 2)

    torch.set_num_threads(1)
    nombre = nombre_corrida(args)
    out = os.path.join(args.dir, nombre, f'semilla_{args.semilla}')
    os.makedirs(out, exist_ok=True)
    R = C.Recompensa()

    # etapas a correr
    if args.especialista:
        etapas = [dict(nombre=f'especialista_{args.especialista}', pasos=2_750_000,
                       escenario=C.ESCENARIOS[args.especialista])]
    elif args.mezcla_desde_cero:
        etapas = [dict(nombre='mezcla', pasos=2_750_000, escenario=C.CURRICULO[-1]['escenario'])]
    else:
        etapas = C.CURRICULO
        if args.etapa:
            etapas = etapas[args.etapa - 1:args.etapa]

    with open(os.path.join(out, 'config.json'), 'w') as fh:
        json.dump(dict(args=vars(args), ppo={k: str(v) for k, v in PPO_KW.items()},
                       recompensa=dataclasses.asdict(R),
                       etapas=[dict(nombre=e['nombre'], pasos=e['pasos'],
                                    escenario=dataclasses.asdict(e['escenario']))
                               for e in etapas]), fh, indent=1)

    modelo, lambdas = None, (0.1, 0.1)
    # si se pide una etapa > 1, partir de la anterior
    i0 = C.CURRICULO.index(etapas[0]) if etapas[0] in C.CURRICULO else 0
    if i0 > 0:
        prev = C.CURRICULO[i0 - 1]['nombre']
        ruta = os.path.join(out, prev + '.zip')
        if not os.path.exists(ruta):
            sys.exit(f'Falta {ruta}: corre antes la etapa {i0}.')
        with open(os.path.join(out, prev + '_lambdas.json')) as fh:
            lambdas = tuple(json.load(fh)['lambdas'])
        modelo = ruta

    for et in etapas:
        pasos = int(et['pasos'] * args.escala_pasos)
        cfg = hacer_cfg(args, et['escenario'])
        envs = hacer_envs(cfg, args.n_envs, args.semilla)
        if modelo is None:
            m = PPO('MlpPolicy', envs, seed=args.semilla, verbose=0, device='cpu',
                    tensorboard_log=os.path.join(args.dir, 'tb') if _hay_tb() else None,
                    **PPO_KW)
        elif isinstance(modelo, str):
            m = PPO.load(modelo, env=envs, device='cpu')
        else:
            m = modelo
            m.set_env(envs)
        cb = CallbackLagrangiano((R.presupuesto_exp, R.presupuesto_soc), lambdas0=lambdas,
                                 actualizar=not args.sin_lagrangiano,
                                 ruta_csv=os.path.join(out, 'progreso.csv'), verbose=1)
        print(f'== {nombre} semilla {args.semilla}: {et["nombre"]} ({pasos} pasos, '
              f'{args.n_envs} entornos)')
        t0 = time.time()
        m.learn(total_timesteps=pasos, callback=cb, reset_num_timesteps=False,
                tb_log_name=f'{nombre}_s{args.semilla}')
        m.save(os.path.join(out, et['nombre'] + '.zip'))
        cb.guardar(os.path.join(out, et['nombre'] + '_lambdas.json'))
        lambdas = tuple(cb.lam)
        modelo = m
        envs.close()
        print(f'   listo en {(time.time() - t0) / 60:.1f} min -> {out}')
    # el último modelo también como final.zip (es el que usa evaluar.py)
    modelo.save(os.path.join(out, 'final.zip'))


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    main()
