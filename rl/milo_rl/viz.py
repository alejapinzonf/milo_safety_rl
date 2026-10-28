"""Dibujos del simulador 2D (para revisar episodios y para figuras del paper)."""
import math

import matplotlib

matplotlib.use('Agg')
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402
from .geometria import Caja, Capsula, Circulo, a_mundo, corta_plano  # noqa: E402

COL = {'muro': '#6b6b6b', 'mueble': '#b08a62', 'mueble_obj': '#8a5a2b', 'objeto': '#d0402c',
       'caja_oculta': '#1e9e6a'}


def _forma(ax, f, **kw):
    if isinstance(f, Circulo):
        ax.add_patch(mp.Circle(f.c, f.r, **kw))
    elif isinstance(f, Caja):
        ax.add_patch(mp.Polygon(f.esquinas(), closed=True, **kw))
    elif isinstance(f, Capsula):
        ax.plot([f.p0[0], f.p1[0]], [f.p0[1], f.p1[1]], lw=f.r * 2 * 72 / 0.9,
                solid_capstyle='round', **{k: v for k, v in kw.items() if k in ('color', 'alpha')})


def dibujar(env, ax=None, titulo=None):
    e = env.esc
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor('#f3ead8')
    for f in env.estaticas:
        alto = f.z[0] <= C.LIDAR_Z <= f.z[1]
        _forma(ax, f, color=COL.get(f.tipo, '#999'), alpha=0.9 if alto else 0.35)
    # objeto
    ob = e.objeto
    ax.plot(*ob['pos'], marker='*', ms=14, color='#e0302c', mec='k', zorder=6)
    ax.plot(*env.obj_est, marker='x', ms=7, color='k', zorder=6)
    # aproximación y salida
    ax.plot(*e.aprox['W'], marker='o', ms=5, color='#2f6fd6', mfc='none')
    ax.plot(*e.salida, marker='s', ms=8, color='#3a9a4a', mfc='none', mew=2)
    if env.camino:
        c = np.array(env.camino)
        ax.plot(c[:, 0], c[:, 1], ':', color='#2f6fd6', lw=1)
    # lidar
    o = a_mundo(env.pose, [C.LIDAR_X, 0.0])
    dirs = env._dirs_loc @ np.array([[math.cos(env.pose[2]), math.sin(env.pose[2])],
                                     [-math.sin(env.pose[2]), math.cos(env.pose[2])]])
    for k in range(0, C.LIDAR_RAYOS, 2):
        if env.tapado[k]:
            d = env.d_mem[k] if env.d_mem[k] < C.LIDAR_RANGO else 0.3
            ax.plot(*zip(o, o + dirs[k] * d), color='#d0402c', lw=0.6, alpha=0.7)
        else:
            ax.plot(*zip(o, o + dirs[k] * env.rangos[k]), color='#4a7bd0', lw=0.3, alpha=0.35)
    # personas
    for h in e.humanos:
        for f in h.formas():
            _forma(ax, f, color='#7a3fb0', alpha=0.85 if f.r < 0.1 else 0.25)
        if h.camina:
            ax.plot(*zip(*h.puntos), '--', color='#7a3fb0', lw=0.8)
    for pos, vel, edad, _ in env.det.estimaciones(env.t):
        if edad < 0.5:
            ax.add_patch(mp.Circle(pos, 0.3, fill=False, ec='#7a3fb0', ls=':'))
    # robot
    base, cap = env._formas_robot()
    _forma(ax, base, color='#2f6fd6', alpha=0.8)
    _forma(ax, cap, color='#e8822a', alpha=0.9)
    ax.arrow(env.pose[0], env.pose[1], 0.3 * math.cos(env.pose[2]), 0.3 * math.sin(env.pose[2]),
             width=0.02, color='k')
    ax.set_xlim(-0.2, e.ancho + 0.2)
    ax.set_ylim(-0.2, e.alto + 0.2)
    ax.set_aspect('equal')
    th = env.th
    ax.set_title(titulo or (f't={env.t:4.1f}s  fase={env.fase}  brazo={env.brazo.estado}\n'
                            f'd_stop={th.d_stop_f:.2f}/{th.d_stop_s:.2f}  v_max={th.v_max:.2f}  '
                            f'v={env.v:.2f}'), fontsize=8)
    ax.tick_params(labelsize=6)
    return ax


def gif(frames, ruta, ms=120):
    from PIL import Image
    ims = [Image.open(f) for f in frames]
    ims[0].save(ruta, save_all=True, append_images=ims[1:], duration=ms, loop=0)


__all__ = ['dibujar', 'gif', 'corta_plano']
