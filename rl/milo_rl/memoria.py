"""Memoria del punto ciego: lo que el lidar vio hace poco, para los rayos que hoy tapa el brazo.

Se guardan los scans recientes en el frame de ODOMETRÍA (que se desvía con el tiempo). Cuando el
brazo tapa un sector, para cada rayo tapado se busca la observación más reciente en esa dirección
y se pone un punto virtual a la distancia CERTIFICADA libre:
    d_k = rango_visto - r_u - e_h
    r_u = sigma0 + k_s * (lo recorrido desde que se vio) + k_psi * |lo girado| * distancia
          (incertidumbre de la odometría: crece con lo que se movió el robot)
    e_h = V_HUMANO_MAX * (t - t_visto)   SOLO si hay personas moviéndose ("erosión": una persona
          pudo haber caminado hasta ahí desde que se miró por última vez)
Si una dirección tapada no tiene NADA en memoria, se supone un obstáculo pegado al robot.
Con esto, la Proposición 1 (no chocar muebles) vale también en los sectores ciegos.
"""
import math

import numpy as np

from . import config as C


class Memoria:
    def __init__(self, params):
        self.p = params
        self.scans = []

    def reset(self):
        self.scans = []

    def guardar(self, pts_odom, t, s, psi):
        if len(pts_odom) == 0:
            return
        self.scans.append(dict(pts=np.asarray(pts_odom), t=t, s=s, psi=psi))
        while self.scans and t - self.scans[0]['t'] > self.p.mem_ventana:
            self.scans.pop(0)

    def virtuales(self, pose_odom, idx_tapados, t, s, psi, dinamico, punta_x):
        """Un punto virtual por rayo tapado, a la distancia CERTIFICADA libre en esa dirección.

        idx_tapados: índices (0..LIDAR_RAYOS-1) de los rayos tapados; el rayo k apunta al ángulo
        -pi + k * 2pi/N (frame robot, desde el lidar). Para cada uno se usa la observación MÁS
        RECIENTE en esa dirección (o en la de al lado):
            d_k = rango_visto - r_u - e_h        (nunca menos que pegado al robot)
        -> (pts (M,2) frame robot, d_k (M,), edad (M,))
        """
        idx = np.asarray(idx_tapados, int)
        if len(idx) == 0:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        N = C.LIDAR_RAYOS
        bin_w = 2 * math.pi / N
        o_lidar = np.array([C.LIDAR_X, 0.0])
        d_pegado = max(punta_x - C.LIDAR_X, 0.0) + 0.01
        d_k = np.full(len(idx), d_pegado)
        edad_k = np.full(len(idx), self.p.mem_ventana)
        if self.scans:
            P = np.concatenate([sc['pts'] for sc in self.scans])
            n_s = [len(sc['pts']) for sc in self.scans]
            SI = np.repeat(np.arange(len(self.scans)), n_s)
            c, sn = math.cos(pose_odom[2]), math.sin(pose_odom[2])
            d = P - np.array(pose_odom[:2])
            loc = np.stack([d[:, 0] * c + d[:, 1] * sn, -d[:, 0] * sn + d[:, 1] * c], axis=1)
            rel = loc - o_lidar
            k = np.round((np.arctan2(rel[:, 1], rel[:, 0]) + math.pi) / bin_w).astype(int) % N
            rango = np.hypot(rel[:, 0], rel[:, 1])
            ultimo = np.full(N, -1)
            np.maximum.at(ultimo, k, SI)
            for j, kk in enumerate(idx):
                cands = [b for b in (kk, (kk - 1) % N, (kk + 1) % N) if ultimo[b] >= 0]
                if not cands:
                    continue
                si = max(ultimo[b] for b in cands)
                m = (SI == si) & np.isin(k, cands)
                sc = self.scans[si]
                r_mem = float(rango[m].min())
                edad = t - sc['t']
                r_u = (self.p.mem_sigma0 + self.p.mem_k_s * abs(s - sc['s'])
                       + self.p.mem_k_psi * abs(float(_ang(psi - sc['psi']))) * r_mem)
                e_h = C.V_HUMANO_MAX * edad if dinamico else 0.0
                d_k[j] = max(d_pegado, r_mem - r_u - e_h)
                edad_k[j] = edad
        a = -math.pi + idx * bin_w
        u = np.stack([np.cos(a), np.sin(a)], axis=1)
        return o_lidar + u * d_k[:, None], d_k, edad_k


def _ang(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
