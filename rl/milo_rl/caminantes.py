"""Personas que caminan de ida y vuelta por una ruta (para la validación en Gazebo).

Sin ROS: dado el tiempo, devuelve dónde está la persona, a qué velocidad va y hacia dónde mira.
Así se prueba aquí sin simulador. La ruta se recorre P0 -> ... -> Pn, se espera `pausa` s,
se vuelve Pn -> ... -> P0, se espera `pausa` s, y así.
"""
import math

import numpy as np


class Ruta:
    def __init__(self, puntos, vel=0.8, pausa=1.0, fase=0.0):
        self.p = np.asarray(puntos, float)
        seg = np.diff(self.p, axis=0)
        self.largo = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.largo)])
        self.L = float(self.cum[-1])
        self.vel, self.pausa, self.fase = float(vel), float(pausa), float(fase)
        self.t_trayecto = self.L / self.vel
        self.periodo = 2 * (self.t_trayecto + self.pausa)

    def _en(self, s):
        """Posición y dirección unitaria a distancia s (0..L) a lo largo de la ruta."""
        s = min(max(s, 0.0), self.L)
        k = min(int(np.searchsorted(self.cum, s, side='right')) - 1, len(self.largo) - 1)
        u = (self.p[k + 1] - self.p[k]) / max(self.largo[k], 1e-9)
        return self.p[k] + u * (s - self.cum[k]), u

    def estado(self, tau):
        """tau = segundos desde que empieza a caminar -> (pos[2], vel[2], yaw)."""
        tau = (tau + self.fase) % self.periodo
        tp = self.t_trayecto
        if tau < tp:
            pos, u = self._en(tau * self.vel)
            return pos, u * self.vel, math.atan2(u[1], u[0])
        if tau < tp + self.pausa:
            pos, u = self._en(self.L)
            return pos, np.zeros(2), math.atan2(u[1], u[0])
        if tau < 2 * tp + self.pausa:
            pos, u = self._en(self.L - (tau - tp - self.pausa) * self.vel)
            return pos, -u * self.vel, math.atan2(-u[1], -u[0])
        pos, u = self._en(0.0)
        return pos, np.zeros(2), math.atan2(-u[1], -u[0])
