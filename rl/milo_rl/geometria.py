"""Geometría 2D: formas con altura, rayos del lidar y distancias/choques.

Cada obstáculo es una forma 2D (círculo, caja girada o cápsula) más un rango de alturas [z0, z1].
- El lidar barre UN plano (z = LIDAR_Z): solo ve las formas cuyo rango de alturas lo corta.
- Para choques se compara el rango de alturas de cada parte del robot (base, brazo) con el del
  obstáculo: el brazo a 0.8 m de alto pasa por encima de una mesa de 0.75 m, la base no.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Circulo:
    c: np.ndarray
    r: float
    z: tuple
    tipo: str = 'mueble'
    ident: int = -1


@dataclass
class Caja:
    c: np.ndarray
    half: tuple
    yaw: float
    z: tuple
    tipo: str = 'mueble'
    ident: int = -1

    def esquinas(self):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        hx, hy = self.half
        loc = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
        rot = np.array([[cy, -sy], [sy, cy]])
        return loc @ rot.T + self.c

    def segmentos(self):
        e = self.esquinas()
        return np.stack([e, np.roll(e, -1, axis=0)], axis=1)


@dataclass
class Capsula:
    p0: np.ndarray
    p1: np.ndarray
    r: float
    z: tuple
    tipo: str = 'brazo'
    ident: int = -1


def corta_plano(forma, z):
    return forma.z[0] <= z <= forma.z[1]


def se_solapan_z(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def rayos_segmentos(o, dirs, segs):
    """Distancia de cada rayo (o + t*dir) a cada segmento. dirs (N,2), segs (M,2,2) -> (N,M)."""
    if len(segs) == 0:
        return np.full((len(dirs), 0), np.inf)
    a = segs[:, 0, :]
    e = segs[:, 1, :] - a
    w = a - o
    den = dirs[:, 0:1] * (-e[None, :, 1]) - dirs[:, 1:2] * (-e[None, :, 0])
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (w[None, :, 0] * (-e[None, :, 1]) - w[None, :, 1] * (-e[None, :, 0])) / den
        u = (dirs[:, 0:1] * w[None, :, 1] - dirs[:, 1:2] * w[None, :, 0]) / den
    ok = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
    return np.where(ok, t, np.inf)


def rayos_circulos(o, dirs, cs, rs):
    """Distancia de cada rayo a cada círculo. cs (K,2), rs (K,) -> (N,K)."""
    if len(cs) == 0:
        return np.full((len(dirs), 0), np.inf)
    w = o - cs
    b = dirs @ w.T
    c = np.sum(w * w, axis=1) - rs ** 2
    disc = b * b - c[None, :]
    with np.errstate(invalid='ignore'):
        sq = np.sqrt(np.maximum(disc, 0.0))
    t1, t2 = -b - sq, -b + sq
    t = np.where(t1 > 0, t1, np.where(t2 > 0, 0.0, np.inf))
    return np.where(disc >= 0, t, np.inf)


class Trazador:
    """Prepara las formas que corta el plano del lidar para tirar rayos rápido (vectorizado)."""

    def __init__(self, formas, z):
        segs, seg_id, cs, rs, c_id = [], [], [], [], []
        self.formas = []
        for f in formas:
            if not corta_plano(f, z):
                continue
            k = len(self.formas)
            self.formas.append(f)
            if isinstance(f, Circulo):
                cs.append(f.c)
                rs.append(f.r)
                c_id.append(k)
            elif isinstance(f, Caja):
                for s in f.segmentos():
                    segs.append(s)
                    seg_id.append(k)
            elif isinstance(f, Capsula):
                for s in _capsula_caja(f).segmentos():
                    segs.append(s)
                    seg_id.append(k)
                cs.append(f.p1)
                rs.append(f.r)
                c_id.append(k)
        self.segs = np.array(segs).reshape(-1, 2, 2)
        self.seg_id = np.array(seg_id, dtype=int)
        self.cs = np.array(cs).reshape(-1, 2)
        self.rs = np.array(rs)
        self.c_id = np.array(c_id, dtype=int)

    def tirar(self, o, dirs, rango):
        """-> (distancia (N,), índice de la forma tocada en self.formas o -1)."""
        ts = rayos_segmentos(o, dirs, self.segs)
        tc = rayos_circulos(o, dirs, self.cs, self.rs)
        t = np.concatenate([ts, tc], axis=1)
        ids = np.concatenate([self.seg_id, self.c_id])
        if t.shape[1] == 0:
            return np.full(len(dirs), rango), np.full(len(dirs), -1)
        j = np.argmin(t, axis=1)
        d = t[np.arange(len(dirs)), j]
        hit = d < rango
        return np.where(hit, d, rango), np.where(hit, ids[j], -1)


def _capsula_caja(cap):
    v = cap.p1 - cap.p0
    ln = float(np.hypot(*v))
    yaw = math.atan2(v[1], v[0]) if ln > 1e-9 else 0.0
    return Caja((cap.p0 + cap.p1) / 2, (ln / 2, cap.r), yaw, cap.z, cap.tipo)


def dist_punto_segmento(p, a, b):
    ab = b - a
    den = float(ab @ ab)
    t = 0.0 if den < 1e-12 else float(np.clip((p - a) @ ab / den, 0.0, 1.0))
    return float(np.hypot(*(a + t * ab - p)))


def _seg_cruza(a, b, c, d):
    def cr(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1, d2, d3, d4 = cr(c, d, a), cr(c, d, b), cr(a, b, c), cr(a, b, d)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def dist_segmento_segmento(a, b, c, d):
    if _seg_cruza(a, b, c, d):
        return 0.0
    return min(dist_punto_segmento(a, c, d), dist_punto_segmento(b, c, d),
               dist_punto_segmento(c, a, b), dist_punto_segmento(d, a, b))


def punto_en_caja(p, caja):
    cy, sy = math.cos(caja.yaw), math.sin(caja.yaw)
    q = p - caja.c
    lx, ly = q[0] * cy + q[1] * sy, -q[0] * sy + q[1] * cy
    return abs(lx) <= caja.half[0] and abs(ly) <= caja.half[1]


def dist_punto_caja(p, caja):
    cy, sy = math.cos(caja.yaw), math.sin(caja.yaw)
    q = p - caja.c
    lx, ly = q[0] * cy + q[1] * sy, -q[0] * sy + q[1] * cy
    dx, dy = max(abs(lx) - caja.half[0], 0.0), max(abs(ly) - caja.half[1], 0.0)
    return math.hypot(dx, dy)


def dist_segmento_caja(a, b, caja):
    if punto_en_caja(a, caja) or punto_en_caja(b, caja):
        return 0.0
    return min(dist_segmento_segmento(a, b, s[0], s[1]) for s in caja.segmentos())


def dist_caja_caja(c1, c2):
    if any(punto_en_caja(p, c2) for p in c1.esquinas()) or \
       any(punto_en_caja(p, c1) for p in c2.esquinas()):
        return 0.0
    return min(dist_segmento_segmento(s[0], s[1], t[0], t[1])
               for s in c1.segmentos() for t in c2.segmentos())


def distancia(a, b):
    """Distancia 2D entre dos formas (0 si se tocan). Ignora alturas."""
    if isinstance(a, Capsula) or isinstance(b, Capsula):
        cap, otra = (a, b) if isinstance(a, Capsula) else (b, a)
        if isinstance(otra, Circulo):
            d = dist_punto_segmento(otra.c, cap.p0, cap.p1) - otra.r
        elif isinstance(otra, Caja):
            d = dist_segmento_caja(cap.p0, cap.p1, otra)
        else:
            d = dist_segmento_segmento(cap.p0, cap.p1, otra.p0, otra.p1) - otra.r
        return max(d - cap.r, 0.0)
    if isinstance(a, Circulo) and isinstance(b, Circulo):
        return max(float(np.hypot(*(a.c - b.c))) - a.r - b.r, 0.0)
    if isinstance(a, Circulo) or isinstance(b, Circulo):
        cir, caja = (a, b) if isinstance(a, Circulo) else (b, a)
        return max(dist_punto_caja(cir.c, caja) - cir.r, 0.0)
    return dist_caja_caja(a, b)


def chocan(a, b):
    """¿Se tocan? (alturas + planta). Versión rápida (separación de ejes), sin numpy."""
    if not se_solapan_z(a.z, b.z):
        return False
    pa = _partes(a) if isinstance(a, Capsula) else [a]
    pb = _partes(b) if isinstance(b, Capsula) else [b]
    return any(_toca(x, y) for x in pa for y in pb)


def _partes(cap):
    return [_capsula_caja(cap), Circulo(cap.p1, cap.r, cap.z), Circulo(cap.p0, cap.r, cap.z)]


def _toca(a, b):
    if isinstance(a, Circulo) and isinstance(b, Circulo):
        dx, dy = a.c[0] - b.c[0], a.c[1] - b.c[1]
        return dx * dx + dy * dy <= (a.r + b.r) ** 2
    if isinstance(a, Circulo) or isinstance(b, Circulo):
        cir, caja = (a, b) if isinstance(a, Circulo) else (b, a)
        cy, sy = math.cos(caja.yaw), math.sin(caja.yaw)
        qx, qy = cir.c[0] - caja.c[0], cir.c[1] - caja.c[1]
        lx, ly = qx * cy + qy * sy, -qx * sy + qy * cy
        dx, dy = max(abs(lx) - caja.half[0], 0.0), max(abs(ly) - caja.half[1], 0.0)
        return dx * dx + dy * dy <= cir.r * cir.r
    return _sat(a, b)


def _sat(a, b):
    """Dos cajas giradas: se tocan si no hay eje que las separe (4 ejes)."""
    dx, dy = b.c[0] - a.c[0], b.c[1] - a.c[1]
    ejes_a = ((math.cos(a.yaw), math.sin(a.yaw)), (-math.sin(a.yaw), math.cos(a.yaw)))
    ejes_b = ((math.cos(b.yaw), math.sin(b.yaw)), (-math.sin(b.yaw), math.cos(b.yaw)))
    for ux, uy in ejes_a + ejes_b:
        ra = a.half[0] * abs(ejes_a[0][0] * ux + ejes_a[0][1] * uy) + \
            a.half[1] * abs(ejes_a[1][0] * ux + ejes_a[1][1] * uy)
        rb = b.half[0] * abs(ejes_b[0][0] * ux + ejes_b[0][1] * uy) + \
            b.half[1] * abs(ejes_b[1][0] * ux + ejes_b[1][1] * uy)
        if abs(dx * ux + dy * uy) > ra + rb:
            return False
    return True


def rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def a_mundo(pose, p_local):
    """Punto(s) del frame del robot al mundo. pose = (x, y, yaw)."""
    return np.asarray(p_local) @ rot(pose[2]).T + np.array(pose[:2])


def a_local(pose, p_mundo):
    return (np.asarray(p_mundo) - np.array(pose[:2])) @ rot(pose[2])


def ang_norm(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
