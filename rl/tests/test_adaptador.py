"""El adaptador a Gazebo da lo mismo que el entorno 2D cuando recibe el mismo scan."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from milo_rl import config as C  # noqa: E402
from milo_rl.adaptador import Adaptador, Persona  # noqa: E402
from milo_rl.baselines import BASELINES  # noqa: E402
from milo_rl.entorno import MiloEnv  # noqa: E402
from milo_rl.escudo import filtrar  # noqa: E402


def _adaptador_de(env, con_personas=False):
    ob = env.esc.objeto
    objeto = dict(pos=np.array(ob['pos'], float), z=ob['z'], donde=ob['donde'])
    pers = [Persona(h.id, h.pos) for h in env.esc.humanos] if con_personas else []
    ad = Adaptador(objeto, pers, semilla=1)
    ad.obj_est = env.obj_est.copy()
    return ad


def _alimentar(ad, env):
    ad.alimentar(env.t, env.pose, env.v, env.w, env._ang, env.rangos, rango_max=C.LIDAR_RANGO)
    # lo que en Gazebo trae el ejecutivo / el nodo, no el sensor:
    ad.fase, ad.cmd, ad.accion_prev = env.fase, env.cmd, env.accion_prev
    ad.info_f, ad.th, ad.t = env.info_f, env.th, env.t


def test_adaptador_igual_al_entorno_sin_personas():
    env = MiloEnv(C.ConfigEntorno(), semilla=0)
    pol = BASELINES['reglas']
    for ep in range(3):
        env.reset(seed=500 + ep)
        ad = _adaptador_de(env)
        n = comparados = 0
        while env.fase in ('navegar', 'orientar', 'acercar') and n < 400:
            env.step(pol(env))
            n += 1
            if env.fase not in ('navegar', 'orientar', 'acercar'):
                break
            _alimentar(ad, env)
            assert np.allclose(np.sort(ad.pts_obst, axis=0), np.sort(env.pts_obst, axis=0),
                               atol=1e-6), 'obstáculos distintos'
            assert np.abs(ad._obs() - env._obs()).max() < 1e-4
            comparados += 1
            v1, w1 = ad.filtrar_cmd(*env.cmd)
            v2, w2, _ = filtrar(*env.cmd, env.th, env.pts_obst, env.brazo, env._personas_local(),
                                env.cfg.escudo, env.det.hay_movimiento())
            assert abs(v1 - v2) < 1e-9 and abs(w1 - w2) < 1e-9
        assert comparados > 20


def test_adaptador_detecta_a_las_personas_que_el_lidar_ve():
    env = MiloEnv(C.ConfigEntorno(escenario=C.ESCENARIOS['quietas']), semilla=0)
    pol = BASELINES['reglas']
    env.reset(seed=7)
    ad = _adaptador_de(env, con_personas=True)
    for _ in range(60):
        _, _, te, tr, _ = env.step(pol(env))
        if te or tr:
            break
        _alimentar(ad, env)
    vistas_env = {i for i, tr in env.det.tracks.items() if tr['n'] >= 3}
    vistas_ad = {i for i, tr in ad.det.tracks.items()}
    assert vistas_env <= vistas_ad


def _correr_mision(modo, semilla_escena, pol=None, escenario='sin_personas'):
    """La misión completa del nodo de Gazebo, con MiloEnv como 'Gazebo' (física + lidar)."""
    import math

    from milo_rl.mision import Mision
    env = MiloEnv(C.ConfigEntorno(escenario=C.ESCENARIOS[escenario]), semilla=0)
    env.reset(seed=semilla_escena)
    ob = env.esc.objeto
    objeto = dict(pos=np.array(ob['pos'], float), z=ob['z'], donde=ob['donde'])
    W = env.esc.aprox['W']
    yaw = math.atan2(objeto['pos'][1] - W[1], objeto['pos'][0] - W[0])
    pers = [Persona(h.id, h.pos) for h in env.esc.humanos]
    ad = Adaptador(objeto, pers, semilla=1, modo=modo)
    mis = Mision(ad, [('obj', objeto, (W[0], W[1], yaw))], politica=pol)
    nav_ok = False
    n = 0
    while not mis.terminada and env.t < 300:
        v_des = w_des = 0.0
        if mis.estado == 'navegando' and env.fase == 'navegar':
            v_des, w_des = env._ejecutivo()
        if mis.estado == 'navegando' and env.fase != 'navegar' and not nav_ok:
            nav_ok = True
            mis.nav_resultado(True)
        sal = mis.paso(env.t, env.pose, env.v, env.w, env._ang, env.rangos, (v_des, w_des),
                       rango_max=C.LIDAR_RANGO)
        env._planta(sal['v'], sal['w'])
        env.t += C.DT
        for h in env.esc.humanos:
            h.paso(C.DT, env._bloquea_humano)
        env._scan()
        n += 1
    return mis.filas


def test_mision_completa_reglas_y_ppo_dummy():
    filas = _correr_mision('reglas', 11)
    assert len(filas) == 1 and filas[0]['resultado'] == 'exito', filas
    assert filas[0]['margen_final'] >= -0.03 and filas[0]['choque'] == 0
    # con una "política" fija a mitad de rango
    filas = _correr_mision('ppo', 12, pol=lambda o: np.zeros(5))
    assert filas[0]['resultado'] == 'exito', filas
    assert filas[0]['d_min_mueble'] > 0.0


def test_mision_con_personas_quietas():
    filas = _correr_mision('reglas', 21, escenario='quietas')
    assert filas[0]['choque'] == 0 and filas[0]['resultado'] in ('exito', 'falla_acercar')
