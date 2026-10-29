"""Pruebas del simulador 2D y del escudo. Desde rl/:  python3 -m pytest -q tests"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from milo_rl import config as C  # noqa: E402
from milo_rl.brazo import Brazo, limites_profundidad, margen_alcance  # noqa: E402
from milo_rl.entorno import N_OBS, MiloEnv  # noqa: E402
from milo_rl.escena import generar  # noqa: E402
from milo_rl.escudo import Theta, filtrar, rampa, v_cap  # noqa: E402
from milo_rl.geometria import (Caja, Capsula, Circulo, Trazador, chocan,  # noqa: E402
                               distancia)


def test_rayo_contra_caja_y_circulo():
    caja = Caja(np.array([2.0, 0.0]), (0.5, 0.5), 0.0, (0, 1))
    cir = Circulo(np.array([0.0, 3.0]), 0.5, (0, 1))
    tr = Trazador([caja, cir], 0.5)
    d, _ = tr.tirar(np.zeros(2), np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), 6.0)
    assert d == pytest.approx([1.5, 2.5, 6.0], abs=1e-6)


def test_choque_rapido_igual_a_distancia_exacta():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        def forma():
            k, c = rng.integers(3), rng.uniform(-1, 1, 2)
            if k == 0:
                return Circulo(c, rng.uniform(0.05, 0.5), (0, 1))
            if k == 1:
                return Caja(c, tuple(rng.uniform(0.05, 0.6, 2)), rng.uniform(-3, 3), (0, 1))
            return Capsula(c, c + rng.uniform(-0.6, 0.6, 2), rng.uniform(0.03, 0.1), (0, 1))
        a, b = forma(), forma()
        assert chocan(a, b) == (distancia(a, b) <= 1e-9)


def test_alturas_separan():
    a = Caja(np.zeros(2), (1, 1), 0, (0.0, 0.37))
    b = Caja(np.zeros(2), (1, 1), 0, (0.71, 0.75))
    assert not chocan(a, b)


def test_alcance_y_limites():
    lim_nom, lim_piso = limites_profundidad(0.8)
    assert lim_nom == pytest.approx(0.97 - 0.03 - 0.44 - 0.30, abs=0.01)
    assert lim_piso - lim_nom == pytest.approx(0.25, abs=1e-6)
    assert margen_alcance((0.9, 0.0), 0.8) > 0
    assert margen_alcance((1.0, 0.0), 0.8) < 0


def test_brazo_tapa_lidar_en_carga_baja():
    b = Brazo()
    assert not b.tapa_lidar()
    b.alcanzar(np.array([0.8, 0.0]), 0.8)
    for _ in range(60):
        b.paso(0.1)
    assert b.estado == 'carga_baja' and b.tapa_lidar()
    assert b.radio_giro() > 0.6
    assert b.despejar()
    for _ in range(25):
        b.paso(0.1)
    assert b.estado == 'guardado' and not b.tapa_lidar()


def test_rampa_sigmoide():
    assert rampa(0.30, 0.30, 0.70) == 0.0
    assert rampa(0.70, 0.30, 0.70) == 1.0
    assert rampa(0.50, 0.30, 0.70) == pytest.approx(0.5, abs=1e-6)
    xs = np.linspace(0.3, 0.7, 50)
    assert np.all(np.diff([rampa(x, 0.3, 0.7) for x in xs]) >= 0)


@pytest.mark.parametrize('d', [0.05, 0.1, 0.3, 0.8])
def test_v_cap_frena_a_tiempo(d):
    """Proposición 1: desde v_cap(d), reaccionando en T y frenando con a, no pasa de d - d_min."""
    a, T, dmin = 0.8, 0.2, 0.05
    v = v_cap(d, dmin, a, T)
    assert v * T + v * v / (2 * a) <= d - dmin + 1e-9


def test_escudo_no_baja_del_piso():
    b = Brazo()
    pts = np.array([[C.FP_X_MAX + 0.04, 0.0]])
    v, w, info = filtrar(0.4, 0.0, Theta(0.0, 0.0, 0.1, 0.4), pts, b, [], C.ParamsEscudo())
    assert v == 0.0
    v, w, info = filtrar(0.4, 0.0, Theta(0.0, 0.0, 0.1, 0.4), pts, b, [],
                         C.ParamsEscudo(activo=False))
    assert v > 0.0


def test_escudo_persona_iso():
    b = Brazo()
    persona = [(np.array([1.0, 0.0]), np.zeros(2), 0.0, 1.0)]
    v, _, _ = filtrar(0.4, 0.0, Theta(0.05, 0.05, 0.1, 0.4), np.zeros((0, 2)), b, persona,
                      C.ParamsEscudo(), hay_movimiento=True)
    assert v == 0.0
    quieta = [(np.array([1.0, 0.0]), np.zeros(2), 0.0, 0.3)]
    v, _, _ = filtrar(0.4, 0.0, Theta(0.05, 0.05, 0.1, 0.4), np.zeros((0, 2)), b, quieta,
                      C.ParamsEscudo(), hay_movimiento=False)
    assert 0.0 < v < 0.4
    v, _, _ = filtrar(0.4, 0.0, Theta(0.05, 0.05, 0.1, 0.4), np.zeros((0, 2)), b, persona,
                      C.ParamsEscudo(), hay_movimiento=False)
    assert v == 0.0


def test_entorno_gymnasium():
    from gymnasium.utils.env_checker import check_env
    env = MiloEnv(semilla=0)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (N_OBS,) and np.all(np.isfinite(obs))


@pytest.mark.parametrize('escenario', list(C.ESCENARIOS))
def test_escudo_sin_choques_con_acciones_al_azar(escenario):
    """Prueba empírica de las Proposiciones: con el escudo, acciones al azar no chocan nunca."""
    env = MiloEnv(C.ConfigEntorno(escenario=C.ESCENARIOS[escenario]), semilla=0)
    rng = np.random.default_rng(0)
    for ep in range(8):
        env.reset(seed=500 + ep)
        while True:
            _, _, te, tr, info = env.step(rng.uniform(-1, 1, 5))
            if te or tr:
                break
        assert info['episodio']['motivo'] != 'choque'
        assert info['episodio']['contacto_persona'] == 0


def test_theta_ida_y_vuelta():
    th = Theta(0.12, 0.25, 0.33, 0.27)
    a = MiloEnv.theta_a_accion(th, True)
    env = MiloEnv(semilla=0)
    th2, desp = env.accion_a_theta(a)
    assert desp and math.isclose(th2.d_stop_f, 0.12, abs_tol=1e-6)
    assert math.isclose(th2.v_max, 0.27, abs_tol=1e-6)


def test_escena_de_estres_tiene_caja_oculta_fuera_del_camino():
    """La caja no mapeada queda en la escena (lidar/choques) pero no en la grilla del planificador,
    lejos del inicio del robot."""
    rng = np.random.default_rng(3)
    for _ in range(5):
        e = generar(rng, C.ESTRES['ciego'])
        cajas = [f for f in e.estaticas if f.tipo == 'caja_oculta']
        assert len(cajas) == 1
        assert np.hypot(*(cajas[0].c - np.asarray(e.inicio[:2]))) > 1.2
        assert e.libre(cajas[0].c, 0.0)


def test_memoria_evita_choques_con_el_brazo_tapando():
    """Con el brazo bajo 4 s de más (cono ciego) el escudo con memoria no choca; es la prueba
    de estrés E3b (sin memoria hay choques, ver LEEME)."""
    from milo_rl.baselines import reglas_tarde
    env = MiloEnv(C.ConfigEntorno(escenario=C.ESTRES['ciego']), semilla=0)
    for i in range(6):
        env.reset(seed=100000 + i)
        while True:
            _, _, te, tr, info = env.step(reglas_tarde(env))
            if te or tr:
                break
        assert info['episodio']['motivo'] != 'choque'
