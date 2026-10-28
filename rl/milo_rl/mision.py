"""Misión de validación en Gazebo: ir al pre-agarre de cada objeto (Nav2), orientarse, acercarse
hasta tener el objeto al alcance y retroceder. Es la máquina de estados que corre el nodo ROS, en
Python puro para poder probarla sin Gazebo (tests/test_adaptador.py la corre contra MiloEnv).

El nodo le pasa en cada tick lo que midió y recibe qué mandar:
    sal = mision.paso(t, pose, v, w, angulos, rangos, cmd_nav)
    sal['v'], sal['w']   -> a /cmd_vel (ya pasaron por el filtro + escudo)
    sal['meta']          -> si no es None: mandar ese goal (x, y, yaw) a Nav2
    sal['cancelar']      -> cancelar el goal de Nav2
y avisa con mision.nav_resultado(ok) cuando Nav2 termina.
"""
import math

import numpy as np

from . import config as C

T_DECISION = 0.2  # [s] cada cuánto elige θ la política
T_NAV_MAX = 150.0  # [s] máximo para llegar al pre-agarre
T_ACERCAR_MAX = 60.0  # [s] máximo para orientarse y acercarse
D_RETROCESO = 0.6  # [m] cuánto retrocede al terminar (queda libre para ir al siguiente)

COLUMNAS = ['modo', 'repeticion', 'objeto', 'resultado', 't_total', 't_nav', 'margen_final',
            'exposicion', 'costo_soc', 'd_min_mueble', 'd_min_persona', 'choque', 'intervenciones',
            'giros_bloqueados', 'violaciones_iso', 'intrusiones_iso', 'recorrido']


class Mision:
    def __init__(self, ad, objetivos, politica=None, repeticiones=1):
        """objetivos: lista de (nombre, objeto (dict de la escena), (x, y, yaw) del pre-agarre)."""
        self.ad = ad
        self.pol = politica
        self.cola = [(r, *o) for r in range(repeticiones) for o in objetivos]
        self.filas = []
        self.estado = 'inicio'
        self.t_ult_decision = -1e9
        self.t_prev = None
        self.nav_ok = None
        self.actual = None

    # ------------------------------------------------------------------------------------------
    def nav_resultado(self, ok):
        self.nav_ok = bool(ok)

    @property
    def terminada(self):
        return self.estado == 'fin'

    def paso(self, t, pose, v, w, angulos, rangos, cmd_nav, rango_max=12.0):
        ad = self.ad
        ad.alimentar(t, pose, v, w, angulos, rangos, rango_max=rango_max)
        dt = 0.05 if self.t_prev is None else float(np.clip(t - self.t_prev, 0.01, 0.2))
        self.t_prev = t
        sal = dict(v=0.0, w=0.0, meta=None, cancelar=False)
        if self.estado == 'fin':
            return sal
        if self.estado == 'inicio':
            if not self.cola:
                self.estado = 'fin'
                return sal
            self.actual = self.cola.pop(0)
            rep, nombre, objeto, pre = self.actual
            ad.nueva_pierna(objeto, t)
            self.t_leg = self.t_acercar = t
            self.nav_ok = None
            self.margen_alc = None
            self.estado = 'navegando'
            sal['meta'] = tuple(pre)
        if t - self.t_ult_decision >= T_DECISION:
            ad.politica(self.pol)
            self.t_ult_decision = t
        v_des = w_des = 0.0
        if self.estado == 'navegando':
            v_des, w_des = cmd_nav
            if self.nav_ok is True:
                self.estado, ad.fase, self.t_acercar = 'acercando', 'orientar', t
                self.t_nav = t - self.t_leg
                v_des = w_des = 0.0
            elif self.nav_ok is False or t - self.t_leg > T_NAV_MAX:
                sal['cancelar'] = self.nav_ok is None
                self._cerrar(t, 'falla_nav')
                v_des = w_des = 0.0
        elif self.estado == 'acercando':
            v_des, w_des = ad.ejecutivo_final()
            if ad.alcanzado:
                self.estado, self.t_parado, self.margen_alc = 'agarre', t, ad.margen()
                v_des = w_des = 0.0
            elif t - self.t_acercar > T_ACERCAR_MAX:
                self._cerrar(t, 'falla_acercar')
        elif self.estado == 'agarre':  # el brazo agarraría acá (en Gazebo va bloqueado): 1 s quieto
            if t - self.t_parado > 1.0:
                self.estado, self.pose_ret, self.t_ret = 'retrocediendo', np.array(pose[:2]), t
                ad.fase = 'retroceder'
        elif self.estado == 'retrocediendo':
            v_des = -0.2
            if np.hypot(*(np.array(pose[:2]) - self.pose_ret)) > D_RETROCESO or t - self.t_ret > 12:
                self._cerrar(t, 'exito')
                v_des = 0.0
        sal['v'], sal['w'] = ad.filtrar_cmd(v_des, w_des, dt)
        return sal

    def _cerrar(self, t, resultado):
        ad = self.ad
        rep, nombre, objeto, pre = self.actual
        m = ad.m
        fila = dict(modo=ad.modo, repeticion=rep, objeto=nombre, resultado=resultado,
                    t_total=round(t - self.t_leg, 1),
                    t_nav=round(getattr(self, 't_nav', float('nan')), 1),
                    margen_final=round(self.margen_alc if self.margen_alc is not None else ad.margen(), 3), exposicion=round(m['exposicion'], 3),
                    costo_soc=round(m['costo_soc'], 3),
                    d_min_mueble=round(m['d_min_mueble'], 3),
                    d_min_persona=round(m['d_min_persona'], 3), choque=int(m['choque']),
                    intervenciones=m['intervenciones'], giros_bloqueados=m['giros_bloqueados'],
                    violaciones_iso=m['violaciones_iso'], intrusiones_iso=m['intrusiones_iso'],
                    recorrido=round(m['recorrido'], 2))
        self.filas.append(fila)
        self.estado = 'inicio'
        self.t_nav = float('nan')


def pose_alcance_ok(ad):
    """¿Está el objeto al alcance (margen >= 0) en la pose actual?"""
    return ad.margen() >= 0.0


__all__ = ['Mision', 'COLUMNAS', 'C']
