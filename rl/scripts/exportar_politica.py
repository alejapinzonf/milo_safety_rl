#!/usr/bin/env python3
"""Exporta un modelo PPO entrenado (final.zip) a un .npz de numpy, para correrlo en ROS sin torch.

  python3 scripts/exportar_politica.py runs/generalista/semilla_0/final.zip politica_s0.npz

Comprueba que la salida coincide con model.predict(deterministic=True) en 200 observaciones.
"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from milo_rl.adaptador import PoliticaNumpy  # noqa: E402


def main():
    zip_in, npz_out = sys.argv[1], sys.argv[2]
    m = PPO.load(zip_in, device='cpu')
    sd = {k: v.cpu().numpy() for k, v in m.policy.state_dict().items()}
    np.savez(npz_out, w0=sd['mlp_extractor.policy_net.0.weight'],
             b0=sd['mlp_extractor.policy_net.0.bias'],
             w1=sd['mlp_extractor.policy_net.2.weight'], b1=sd['mlp_extractor.policy_net.2.bias'],
             w2=sd['action_net.weight'], b2=sd['action_net.bias'])
    pol = PoliticaNumpy(npz_out)
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(200):
        o = rng.normal(0, 1, 97).astype(np.float32)
        err = max(err, float(np.abs(pol(o) - m.predict(o, deterministic=True)[0]).max()))
    print(f'{npz_out}: error máximo vs SB3 = {err:.2e}')
    assert err < 1e-4, 'la política en numpy no coincide con SB3'


if __name__ == '__main__':
    main()
