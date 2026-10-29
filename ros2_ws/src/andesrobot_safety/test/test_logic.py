import math

import numpy as np

from andesrobot_safety.logic import SafetyParams, filter_cmd, remove_self

P = SafetyParams()


def wall_at(x, y0=-1.0, y1=1.0, n=50):
    return np.column_stack((np.full(n, x), np.linspace(y0, y1, n)))


def test_libre_avanza_y_satura():
    v, w, s = filter_cmd(1.0, 0.0, np.zeros((0, 2)), P)
    assert v == P.max_linear and s == 'libre'


def test_para_en_stop_distance():
    pts = wall_at(P.footprint_x_max + P.stop_distance - 0.05)
    v, w, s = filter_cmd(0.3, 0.0, pts, P)
    assert v == 0.0 and 'PARADO adelante' in s


def test_frena_en_zona_lenta():
    pts = wall_at(P.footprint_x_max + (P.stop_distance + P.slow_distance) / 2)
    v, _, _ = filter_cmd(0.4, 0.0, pts, P)
    assert math.isclose(v, 0.2, rel_tol=1e-6)


def test_puede_retroceder_con_muro_adelante():
    pts = wall_at(P.footprint_x_max + 0.1)
    v, _, _ = filter_cmd(-0.2, 0.0, pts, P)
    assert v == -0.2


def test_para_atras():
    pts = wall_at(P.footprint_x_min - 0.2)
    v, _, s = filter_cmd(-0.2, 0.0, pts, P)
    assert v == 0.0 and 'atrás' in s


def test_pasa_puerta_de_1m():
    xs = np.linspace(0.5, 2.5, 40)
    pts = np.vstack([np.column_stack((xs, np.full(40, 0.5))), np.column_stack((xs, np.full(40, -0.5)))])
    v, _, _ = filter_cmd(0.3, 0.0, pts, P)
    assert v == 0.3


def test_giro_bloqueado_si_algo_muy_cerca():
    pts = np.array([[0.0, 0.6]])
    _, w, s = filter_cmd(0.0, 0.8, pts, P)
    assert w == 0.0 and 'giro bloqueado' in s


def test_giro_permitido_con_espacio():
    pts = wall_at(P.footprint_x_max + 1.2)
    _, w, _ = filter_cmd(0.0, 0.8, pts, P)
    assert w == 0.8


def test_ignora_columna_propia():
    pts = np.array([[0.0, 0.0], [-0.03, 0.02]])
    assert remove_self(pts, P).size == 0


def test_nav_curva_en_puerta_permitida():
    pn = SafetyParams(turn_check_only_in_place=True)
    xs = np.linspace(-0.3, 1.0, 30)
    pts = np.vstack([np.column_stack((xs, np.full(30, 0.45))), np.column_stack((xs, np.full(30, -0.45)))])
    v, w, _ = filter_cmd(0.2, 0.3, pts, pn)
    assert v == 0.2 and w == 0.3


def test_nav_giro_en_el_lugar_en_puerta_bloqueado():
    pn = SafetyParams(turn_check_only_in_place=True)
    pts = np.array([[0.0, 0.45], [0.0, -0.45]])
    _, w, s = filter_cmd(0.0, 0.5, pts, pn)
    assert w == 0.0 and 'giro bloqueado' in s
