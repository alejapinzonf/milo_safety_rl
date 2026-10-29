"""Generador de escenas aleatorias: una pieza de casa con UN objeto para agarrar.

Cada episodio es un ciclo de agarre (el escenario C del paper, recortado a una pieza):
  Milo parte en un punto al azar -> va frente al mueble -> se acerca -> agarra -> se retira ->
  va a la "salida" (el siguiente lugar de la casa).

Qué se sortea (domain randomization):
  - la pieza (6-8 x 5-7 m), el mueble del objeto (mesa rectangular, redonda, estante o suelo), sus
    medidas y alturas, sillas alrededor, 1-4 muebles extra, contra el muro o no
  - el objeto (botella, taza, galletas), a qué altura y a qué PROFUNDIDAD EFECTIVA: el 25 % se
    alcanza con el filtro nominal y el 75 % solo relajándolo (como en la casa de Gazebo)
  - dónde parte Milo, dónde está la salida
  - las personas: ninguna, 1-3 quietas o 1-3 caminando (según el Escenario)

Igual que en la casa de Gazebo, las mesas y sillas tienen "faldón" a la altura del lidar.
"""
import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import config as C
from .brazo import limites_profundidad
from .geometria import Caja, Circulo, distancia, se_solapan_z
from .humanos import Humano

FALDON_Z = (0.15, 0.45)
NIVELES_ESTANTE = (0.45, 0.85, 1.25, 1.62)
OBJETOS = {
    'botella': dict(alto=0.22, r=0.033),
    'taza': dict(alto=0.095, r=0.08),
    'galletas': dict(alto=0.035, r=0.10),
}
RES = 0.1
INFLADO = 0.32
DIST_PRE = 0.6
GIRO_LIBRE = 0.30


@dataclass
class Escena:
    ancho: float
    alto: float
    estaticas: list
    objeto: dict
    aprox: dict
    inicio: tuple
    salida: np.ndarray
    humanos: list = field(default_factory=list)
    tipo_personas: str = 'sin_personas'
    grilla: np.ndarray = None
    dist_libre: np.ndarray = None

    def celda(self, p):
        return (int(np.clip(p[1] / RES, 0, self.grilla.shape[0] - 1)),
                int(np.clip(p[0] / RES, 0, self.grilla.shape[1] - 1)))

    def libre(self, p, margen=0.0):
        i, j = self.celda(p)
        return self.dist_libre[i, j] > margen

    def planificar(self, a, b, evitar=()):
        """A* en la grilla (8 vecinos). evitar = [(centro, radio)] (personas detectadas, como
        las pone Nav2 en su costmap). Si con ellas no hay camino, planifica sin ellas (el escudo
        lo frena si hace falta). Devuelve una lista de puntos (mundo) o None."""
        if evitar:
            G = self.grilla.copy()
            ny, nx = G.shape
            ys, xs = (np.arange(ny) + 0.5) * RES, (np.arange(nx) + 0.5) * RES
            for c, r in evitar:
                G &= ((xs[None, :] - c[0]) ** 2 + (ys[:, None] - c[1]) ** 2) > r * r
            camino = a_estrella(self, a, b, G)
            if camino is not None:
                return camino
        return a_estrella(self, a, b)


def generar(rng, esc):
    """Devuelve una Escena válida (reintenta hasta que todo calce)."""
    for _ in range(200):
        e = _intentar(rng, esc)
        if e is not None:
            return e
    raise RuntimeError('no se pudo generar una escena válida')


def _intentar(rng, esc):
    W, H = rng.uniform(6.0, 8.0), rng.uniform(5.0, 7.0)
    est = _muros(W, H)
    tipo = rng.choice(list(esc.p_mueble), p=_norm(list(esc.p_mueble.values())))
    r = {'mesa': _mesa, 'redonda': _redonda, 'estante': _estante, 'suelo': _suelo}[tipo](
        rng, esc, W, H)
    if r is None:
        return None
    formas_obj, objeto, aprox = r
    est += formas_obj
    if tipo in ('mesa', 'redonda') and rng.random() < 0.6:
        est += _sillas(rng, objeto['mueble_forma'], aprox, est)
    for _ in range(rng.integers(esc.n_distractores[0], esc.n_distractores[1] + 1)):
        f = _distractor(rng, W, H, est, aprox, objeto)
        if f is not None:
            est += f
    e = Escena(W, H, est, objeto, aprox, None, None)
    _grilla(e)
    if not e.libre(aprox['W'], GIRO_LIBRE):
        return None
    inicio = _punto_libre(rng, e, cerca=aprox['W'], dist=(2.0, 4.5), margen=GIRO_LIBRE)
    if inicio is None or e.planificar(inicio, aprox['W']) is None:
        return None
    salida = _punto_libre(rng, e, cerca=objeto['pos'], dist=(2.0, 3.5), margen=GIRO_LIBRE,
                          lejos_de=[inicio], sep=1.0)
    if salida is None or e.planificar(aprox['W'], salida) is None:
        return None
    e.inicio = (inicio[0], inicio[1], rng.uniform(-math.pi, math.pi))
    e.salida = salida
    if esc.p_oculto > 0 and rng.random() < esc.p_oculto:
        f = _oculto(rng, e)
        if f is None:
            return None
        e.estaticas = e.estaticas + [f]
    u = rng.random()
    if u < esc.p_sin_personas:
        e.tipo_personas = 'sin_personas'
    elif u < esc.p_sin_personas + esc.p_quietas:
        e.tipo_personas = 'quietas'
    else:
        e.tipo_personas = 'caminando'
    if e.tipo_personas != 'sin_personas':
        n = int(rng.integers(esc.n_personas[0], esc.n_personas[1] + 1))
        e.humanos = _personas(rng, e, n, e.tipo_personas == 'caminando', esc)
    return e


def _oculto(rng, e):
    """Caja baja que el planificador no conoce, sobre la ruta del pre-agarre a la salida.

    Queda a >= 0.9 m del pre-agarre y a >= 1.0 m de la salida, lejos del objeto, y solo si se la
    puede rodear (hay otra ruta que la esquive): así el robot la ve antes de agarrar, la pierde
    detrás del brazo al salir, y el episodio sigue siendo posible."""
    ruta = e.planificar(e.aprox['W'], e.salida)
    if ruta is None:
        return None
    pts = np.array(ruta)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    cand = [p for p, si in zip(pts, s) if 0.9 <= si <= s[-1] - 1.0 and e.libre(p, 0.35)
            and np.hypot(*(p - e.objeto['pos'])) > 1.0
            and np.hypot(*(p - np.asarray(e.inicio[:2]))) > 1.2]
    ida = e.planificar(np.asarray(e.inicio[:2]), e.aprox['W'])
    ida = np.array(ida) if ida is not None else np.zeros((0, 2))
    cand = [p for p in cand if len(ida) == 0 or min(np.hypot(*(ida - p).T)) > 0.9]
    rng.shuffle(cand)
    for c in cand[:10]:
        alt = e.planificar(e.aprox['W'], e.salida, [(c, 0.65)])
        if alt is not None and min(np.hypot(*(np.array(alt) - c).T)) >= 0.6:
            return Caja(np.array(c, float), (0.175, 0.175), rng.uniform(0, math.pi / 2),
                        (0.0, 0.5), 'caja_oculta')
    return None


def _muros(W, H):
    t, z = 0.1, (0.0, 1.2)
    return [Caja(np.array([W / 2, -t / 2]), (W / 2 + t, t / 2), 0.0, z, 'muro'),
            Caja(np.array([W / 2, H + t / 2]), (W / 2 + t, t / 2), 0.0, z, 'muro'),
            Caja(np.array([-t / 2, H / 2]), (t / 2, H / 2), 0.0, z, 'muro'),
            Caja(np.array([W + t / 2, H / 2]), (t / 2, H / 2), 0.0, z, 'muro')]


def _contra_muro(rng, W, H, largo, fondo):
    """Centro, yaw y normal hacia adentro de un mueble apoyado en un muro al azar."""
    m = rng.integers(4)
    if m == 0:
        x = rng.uniform(largo / 2 + 0.2, W - largo / 2 - 0.2)
        return np.array([x, fondo / 2]), 0.0, np.array([0.0, 1.0])
    if m == 1:
        x = rng.uniform(largo / 2 + 0.2, W - largo / 2 - 0.2)
        return np.array([x, H - fondo / 2]), 0.0, np.array([0.0, -1.0])
    if m == 2:
        y = rng.uniform(largo / 2 + 0.2, H - largo / 2 - 0.2)
        return np.array([fondo / 2, y]), math.pi / 2, np.array([1.0, 0.0])
    y = rng.uniform(largo / 2 + 0.2, H - largo / 2 - 0.2)
    return np.array([W - fondo / 2, y]), math.pi / 2, np.array([-1.0, 0.0])


def _elegir_objeto(rng, esc):
    tipo = rng.choice(list(OBJETOS))
    return tipo, OBJETOS[tipo], rng.random() < esc.p_facil


def _banda(z, facil):
    lim_nom, lim_piso = limites_profundidad(z)
    if facil:
        return (0.06, lim_nom - 0.03), lim_nom, lim_piso
    return (lim_nom + 0.03, lim_piso - 0.03), lim_nom, lim_piso


def _mesa(rng, esc, W, H):
    L, D, h = rng.uniform(1.0, 1.8), rng.uniform(0.6, 1.0), rng.uniform(0.70, 0.80)
    if rng.random() < 0.4:
        c, yaw, n_muro = _contra_muro(rng, W, H, L, D)
    else:
        yaw = float(rng.choice([0.0, math.pi / 2]))
        ex, ey = (L / 2, D / 2) if yaw == 0.0 else (D / 2, L / 2)
        if W - 2 * (ex + 1.4) < 0 or H - 2 * (ey + 1.4) < 0:
            return None
        c = np.array([rng.uniform(ex + 1.4, W - ex - 1.4), rng.uniform(ey + 1.4, H - ey - 1.4)])
        n_muro = None
    faldon = Caja(c, (L / 2, D / 2), yaw, FALDON_Z, 'mueble_obj')
    cubierta = Caja(c, (L / 2, D / 2), yaw, (h - 0.04, h), 'mueble_obj')
    tipo, ob, facil = _elegir_objeto(rng, esc)
    zc = h + ob['alto'] / 2
    banda, lim_nom, lim_piso = _banda(zc, facil)
    cy, sy = math.cos(yaw), math.sin(yaw)
    ex_, ey_ = np.array([cy, sy]), np.array([-sy, cy])
    bordes = [(ex_, L / 2, D / 2), (-ex_, L / 2, D / 2), (ey_, D / 2, L / 2), (-ey_, D / 2, L / 2)]
    bordes = [b for b in bordes if n_muro is None or float(b[0] @ n_muro) > -0.5]
    for _ in range(30):
        n, semi, semi_lat = bordes[rng.integers(len(bordes))]
        t = np.array([-n[1], n[0]])
        p = rng.uniform(*banda)
        if p < ob['r'] + 0.02 or p > 2 * semi - ob['r'] - 0.02:
            continue
        lat = rng.uniform(-(semi_lat - ob['r'] - 0.05), semi_lat - ob['r'] - 0.05)
        pos = c + n * (semi - p) + t * lat
        prof = [(semi_b - float((pos - c) @ nb), nb) for nb, semi_b, _ in bordes]
        pe, nbest = min(prof, key=lambda x: x[0])
        if banda[0] <= pe <= banda[1]:
            borde = pos + nbest * pe
            return [faldon, cubierta], _obj(tipo, ob, pos, h, pe, lim_nom, lim_piso, 'mesa',
                                            faldon), _aprox(borde, nbest)
    return None


def _redonda(rng, esc, W, H):
    r, h = rng.uniform(0.5, 0.8), rng.uniform(0.70, 0.80)
    m = r + 1.4
    if W - 2 * m < 0 or H - 2 * m < 0:
        return None
    c = np.array([rng.uniform(m, W - m), rng.uniform(m, H - m)])
    faldon = Circulo(c, r, FALDON_Z, 'mueble_obj')
    cubierta = Circulo(c, r, (h - 0.04, h), 'mueble_obj')
    tipo, ob, facil = _elegir_objeto(rng, esc)
    banda, lim_nom, lim_piso = _banda(h + ob['alto'] / 2, facil)
    p = rng.uniform(*banda)
    if p > r - 0.02:
        return None
    phi = rng.uniform(-math.pi, math.pi)
    n = np.array([math.cos(phi), math.sin(phi)])
    pos = c + n * (r - p)
    return [faldon, cubierta], _obj(tipo, ob, pos, h, p, lim_nom, lim_piso, 'redonda',
                                    faldon), _aprox(c + n * r, n)


def _estante(rng, esc, W, H):
    L, D = rng.uniform(0.8, 1.4), rng.uniform(0.32, 0.45)
    c, yaw, n = _contra_muro(rng, W, H, L, D)
    tb = 0.02
    t = np.array([-n[1], n[0]])
    formas = [Caja(c, (L / 2, D / 2), yaw, (0.0, NIVELES_ESTANTE[0]), 'mueble_obj')]
    hz = (NIVELES_ESTANTE[0], 2.0)
    formas.append(Caja(c - n * (D / 2 - tb / 2), (L / 2, tb / 2), yaw, hz, 'mueble_obj'))
    for s in (-1, 1):
        formas.append(Caja(c + t * s * (L / 2 - tb / 2), (tb / 2, D / 2), yaw, hz, 'mueble_obj'))
    for z in NIVELES_ESTANTE[1:] + (2.0,):
        formas.append(Caja(c, (L / 2, D / 2), yaw, (z - tb, z), 'mueble_obj'))
    tipo, ob, facil = _elegir_objeto(rng, esc)
    nivel = float(rng.choice(NIVELES_ESTANTE))
    banda, lim_nom, lim_piso = _banda(nivel + ob['alto'] / 2, facil)
    hi = min(banda[1], D - tb - ob['r'] - 0.01)
    if hi <= banda[0]:
        return None
    p = rng.uniform(banda[0], hi)
    lat = rng.uniform(-(L / 2 - ob['r'] - 0.05), L / 2 - ob['r'] - 0.05)
    borde = c + n * (D / 2) + t * lat
    pos = borde - n * p
    return formas, _obj(tipo, ob, pos, nivel, p, lim_nom, lim_piso, 'estante',
                        formas[0]), _aprox(borde, n)


def _suelo(rng, esc, W, H):
    """Objeto en el suelo delante de un mueble bajo apoyado en el muro (el lidar no lo ve)."""
    L, D, h = rng.uniform(0.8, 2.0), rng.uniform(0.4, 0.9), rng.uniform(0.45, 0.9)
    c, yaw, n = _contra_muro(rng, W, H, L, D)
    mueble = Caja(c, (L / 2, D / 2), yaw, (0.0, h), 'mueble')
    tipo, ob, _ = _elegir_objeto(rng, esc)
    sep = rng.uniform(0.10, 0.30)
    t = np.array([-n[1], n[0]])
    lat = rng.uniform(-(L / 2 - 0.2), L / 2 - 0.2)
    pos = c + n * (D / 2 + sep) + t * lat
    o = _obj(tipo, ob, pos, 0.0, None, None, None, 'suelo', None)
    o['forma'] = Circulo(pos.copy(), min(ob['r'], 0.06), (0.0, ob['alto']), 'objeto')
    borde = pos + n * min(ob['r'], 0.06)
    return [mueble, o['forma']], o, _aprox(borde, n)


def _obj(tipo, ob, pos, z_base, prof, lim_nom, lim_piso, donde, mueble_forma):
    return dict(tipo=tipo, pos=np.array(pos, float), z=z_base + ob['alto'] / 2, z_base=z_base,
                alto=ob['alto'], prof_ef=prof, lim_nom=lim_nom, lim_piso=lim_piso, donde=donde,
                facil=(prof is not None and prof <= lim_nom), mueble_forma=mueble_forma,
                forma=None)


def _aprox(borde, n):
    n = np.asarray(n, float) / np.hypot(*n)
    W = borde + n * (DIST_PRE + C.FP_X_MAX)
    return dict(n=n, borde=np.asarray(borde, float), W=W)


def _cerca_de_linea(p, a, b, d):
    ab = b - a
    t = np.clip((p - a) @ ab / max(ab @ ab, 1e-9), 0, 1)
    return float(np.hypot(*(a + t * ab - p))) < d


def _sillas(rng, mesa, aprox, est):
    out, lado = [], 0.42
    if isinstance(mesa, Circulo):
        k = int(rng.integers(2, 5))
        a0 = rng.uniform(0, 2 * math.pi)
        cands = [(mesa.c + (mesa.r + 0.05) * np.array([math.cos(a), math.sin(a)]), a + math.pi)
                 for a in a0 + np.arange(k) * 2 * math.pi / k]
    else:
        cands = []
        cy, sy = math.cos(mesa.yaw), math.sin(mesa.yaw)
        for s in (-1, 1):
            for f in (-0.3, 0.3):
                for eje, semi, semi_l in ((np.array([-sy, cy]), mesa.half[1], mesa.half[0]),):
                    t = np.array([cy, sy])
                    p = mesa.c + eje * s * (semi + 0.12) + t * f * semi_l
                    cands.append((p, math.atan2(-s * eje[1], -s * eje[0])))
    for p, yaw in cands:
        if rng.random() < 0.3:
            continue
        if _cerca_de_linea(p, aprox['borde'], aprox['W'], 0.75):
            continue
        asiento = Caja(p, (lado / 2, lado / 2), yaw, (0.0, 0.45), 'mueble')
        if any(f.tipo == 'muro' and distancia(asiento, f) < 0.05 for f in est):
            continue
        u = np.array([math.cos(yaw), math.sin(yaw)])
        resp = Caja(p - u * (lado / 2 - 0.02), (0.02, lado / 2), yaw, (0.45, 0.85), 'mueble')
        out += [asiento, resp]
    return out


def _distractor(rng, W, H, est, aprox, objeto):
    for _ in range(20):
        if rng.random() < 0.7:
            L, D = rng.uniform(0.4, 2.0), rng.uniform(0.35, 0.9)
            h = rng.uniform(0.4, 1.9)
            if rng.random() < 0.7:
                c, yaw, _ = _contra_muro(rng, W, H, L, D)
            else:
                c = np.array([rng.uniform(1, W - 1), rng.uniform(1, H - 1)])
                yaw = float(rng.choice([0.0, math.pi / 2]))
            f = Caja(c, (L / 2, D / 2), yaw, (0.0, h), 'mueble')
        else:
            r = rng.uniform(0.12, 0.25)
            c = np.array([rng.uniform(r + 0.1, W - r - 0.1), rng.uniform(r + 0.1, H - r - 0.1)])
            f = Circulo(c, r, (0.0, rng.uniform(0.5, 1.6)), 'mueble')
        if _cerca_de_linea(f.c, aprox['borde'], aprox['W'], 1.2):
            continue
        if any(distancia(f, g) < 0.9 for g in est if g.tipo != 'muro'):
            continue
        if any(g.tipo == 'muro' and 0.0 < distancia(f, g) < 0.85 for g in est):
            continue
        return [f]
    return None


def _grilla(e):
    nx, ny = int(math.ceil(e.ancho / RES)), int(math.ceil(e.alto / RES))
    xs, ys = (np.arange(nx) + 0.5) * RES, (np.arange(ny) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    occ = np.zeros((ny, nx), bool)
    for f in e.estaticas:
        if not se_solapan_z(f.z, (0.0, C.BASE_Z)):
            continue
        if isinstance(f, Circulo):
            occ |= (X - f.c[0]) ** 2 + (Y - f.c[1]) ** 2 <= f.r ** 2
        else:
            cy, sy = math.cos(f.yaw), math.sin(f.yaw)
            dx, dy = X - f.c[0], Y - f.c[1]
            lx, ly = dx * cy + dy * sy, -dx * sy + dy * cy
            occ |= (np.abs(lx) <= f.half[0] + RES / 2) & (np.abs(ly) <= f.half[1] + RES / 2)
    borde = np.zeros_like(occ)
    borde[0, :] = borde[-1, :] = True
    borde[:, 0] = borde[:, -1] = True
    occ |= borde
    e.dist_libre = distance_transform_edt(~occ) * RES - INFLADO
    e.grilla = e.dist_libre > 0


def _punto_libre(rng, e, cerca, dist, margen, lejos_de=(), sep=0.0):
    for _ in range(200):
        a = rng.uniform(-math.pi, math.pi)
        d = rng.uniform(*dist)
        p = np.asarray(cerca) + d * np.array([math.cos(a), math.sin(a)])
        if not (0.3 < p[0] < e.ancho - 0.3 and 0.3 < p[1] < e.alto - 0.3):
            continue
        if not e.libre(p, margen):
            continue
        if any(np.hypot(*(p - np.asarray(q[:2]))) < sep for q in lejos_de):
            continue
        return p
    return None


def a_estrella(e, a, b, G=None):
    s, g = e.celda(a), e.celda(b)
    G = e.grilla if G is None else G
    if not G[g]:
        return None
    if not G[s]:
        ii, jj = np.nonzero(G)
        k = np.argmin((ii - s[0]) ** 2 + (jj - s[1]) ** 2)
        s = (int(ii[k]), int(jj[k]))
    ny, nx = G.shape
    pena = np.clip(0.4 - e.dist_libre, 0, None) * 4
    abiertos = [(0.0, s)]
    costo = {s: 0.0}
    padre = {}
    vec = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414),
           (-1, 1, 1.414), (-1, -1, 1.414)]
    while abiertos:
        _, u = heapq.heappop(abiertos)
        if u == g:
            break
        for di, dj, w in vec:
            v = (u[0] + di, u[1] + dj)
            if not (0 <= v[0] < ny and 0 <= v[1] < nx) or not G[v]:
                continue
            nc = costo[u] + w + pena[v]
            if nc < costo.get(v, 1e18):
                costo[v] = nc
                padre[v] = u
                h = math.hypot(v[0] - g[0], v[1] - g[1])
                heapq.heappush(abiertos, (nc + h, v))
    if g not in costo:
        return None
    camino = [g]
    while camino[-1] != s:
        camino.append(padre[camino[-1]])
    camino.reverse()
    pts = [np.array([(j + 0.5) * RES, (i + 0.5) * RES]) for i, j in camino]
    pts[-1] = np.asarray(b, float)
    return pts[::3] + [pts[-1]] if len(pts) > 3 else pts


def _ok_persona(e, p, margen=0.25):
    ap = e.aprox
    if not e.libre(p, margen):
        return False
    if _cerca_de_linea(p, ap['borde'], ap['W'], 0.9):
        return False
    for q in (ap['W'], e.inicio[:2], e.salida, e.objeto['pos']):
        if np.hypot(*(p - np.asarray(q))) < 1.0:
            return False
    return True


def _personas(rng, e, n, caminan, esc):
    hs = []
    for k in range(n):
        for _ in range(100):
            if not caminan:
                if rng.random() < 0.6:
                    a = rng.uniform(-math.pi, math.pi)
                    p = e.objeto['pos'] + rng.uniform(1.0, 2.2) * np.array([math.cos(a),
                                                                            math.sin(a)])
                else:
                    p = np.array([rng.uniform(0.5, e.ancho - 0.5), rng.uniform(0.5, e.alto - 0.5)])
                if _ok_persona(e, p) and all(np.hypot(*(p - h.pos)) > 0.8 for h in hs):
                    hs.append(Humano(k, p, rng.uniform(-math.pi, math.pi), rng=rng))
                    break
            else:
                A = np.array([rng.uniform(0.5, e.ancho - 0.5), rng.uniform(0.5, e.alto - 0.5)])
                if rng.random() < 0.5:
                    m = (np.asarray(e.inicio[:2]) + e.aprox['W'] + e.salida) / 3
                    B = 2 * m - A + rng.normal(0, 0.5, 2)
                else:
                    B = np.array([rng.uniform(0.5, e.ancho - 0.5), rng.uniform(0.5, e.alto - 0.5)])
                if not (2.0 <= np.hypot(*(B - A)) <= 6.0):
                    continue
                if not (_ok_persona(e, A) and _ok_persona(e, B)):
                    continue
                L = float(np.hypot(*(B - A)))
                if not all(e.libre(A + (B - A) * s, 0.0) for s in np.linspace(0, 1, int(L / 0.1))):
                    continue
                v = rng.uniform(*esc.v_caminar)
                hs.append(Humano(k, A, math.atan2(*(B - A)[::-1]), puntos=[A, B], v=v,
                                 pausa=2.0, rng=rng))
                break
    return hs


def _norm(p):
    p = np.asarray(p, float)
    return p / p.sum()
