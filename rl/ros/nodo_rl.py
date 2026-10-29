#!/usr/bin/env python3
"""Nodo ROS 2 de la validación en Gazebo (E5): el filtro + escudo del paper con el PPO.

Reemplaza al `safety_filter` en la simulación de navegación. Es solo el "pegamento" con ROS: toda
la lógica está en milo_rl/ (adaptador.py y mision.py, probados contra el simulador 2D).

  /scan, TF, /odom ─┐
  /safety/cmd_vel_in (Nav2 vía twist_mux) ─┤  Mision.paso()  ─> /cmd_vel   (filtro + escudo)
  NavigateToPose (Nav2) <─ goal al pre-agarre ─┘

Lo lanza `./sim.sh rl ...` (ver rl/LEEME_GAZEBO.md). Parámetros (ros__parameters):
  modo        ppo | nominal | minimo | reglas
  politica    ruta al .npz de la política (solo modo ppo; ver scripts/exportar_politica.py)
  objetos     nombres de robocup_casa_objetos.yaml separados por coma, en el orden a visitar
  repeticiones  veces que se repite la lista
  objetos_yaml, personas_yaml (opcional), csv (salida), semilla
"""
import csv
import math
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import yaml  # noqa: E402
from geometry_msgs.msg import PoseStamped, Twist  # noqa: E402
from nav2_msgs.action import NavigateToPose  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tf2_ros import Buffer, TransformListener  # noqa: E402

from milo_rl import config as C  # noqa: E402
from milo_rl.adaptador import Adaptador, Persona, PoliticaNumpy, objeto_desde_yaml  # noqa: E402
from milo_rl.caminantes import Ruta  # noqa: E402
from milo_rl.mision import COLUMNAS, Mision  # noqa: E402

try:
    from gazebo_msgs.srv import SetEntityState  # noqa: E402
except ImportError:  # pragma: no cover
    SetEntityState = None


ESPERA_ARRANQUE = 8.0
MAX_REINTENTOS = 8
T_REINTENTO = 3.0


def yaw_de(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NodoRL(Node):
    def __init__(self):
        super().__init__('rl_filtro')
        for k, v in dict(modo='ppo', politica='', objetos='', repeticiones=1,
                         objetos_yaml='', personas_yaml='', csv='', semilla=0,
                         cmd_in='/safety/cmd_vel_in', cmd_out='/cmd_vel',
                         base_frame='base_footprint', map_frame='map').items():
            self.declare_parameter(k, v)
        def P(k):
            v = self.get_parameter(k).value
            return '' if v == 'ninguna' else v

        self.base, self.mapf = P('base_frame'), P('map_frame')
        modo = P('modo')
        pol = PoliticaNumpy(P('politica')) if modo == 'ppo' else None

        with open(P('objetos_yaml')) as fh:
            objs = yaml.safe_load(fh)
        nombres = [n.strip() for n in P('objetos').split(',') if n.strip()]
        objetivos = []
        for n in nombres:
            d = objs[n]
            if 'pre' not in d:
                raise SystemExit(f'{n}: no tiene pose de pre-agarre (¿objeto en el suelo?)')
            objetivos.append((n, objeto_desde_yaml(n, d), tuple(d['pre'])))
        personas = []
        self.caminantes = []
        rng_fase = np.random.default_rng(int(P('semilla')))
        if P('personas_yaml'):
            with open(P('personas_yaml')) as fh:
                for i, (n, d) in enumerate(yaml.safe_load(fh).items()):
                    if d.get('tipo') == 'caminante':
                        ruta = Ruta(d['ruta'], d.get('vel', 0.8), d.get('pausa', 1.0))
                        ruta.fase = float(rng_fase.uniform(0, ruta.periodo))
                        h = Persona(i, d['ruta'][0])
                        self.caminantes.append((n, h, ruta))
                        personas.append(h)
                    else:
                        personas.append(Persona(i, (d['x'], d['y'])))
        self.get_logger().info(f'modo={modo}, {len(objetivos)} objetos, {len(personas)} personas')
        self.ad = Adaptador(objetivos[0][1], personas, semilla=int(P('semilla')), modo=modo)
        self.mision = Mision(self.ad, objetivos, pol, int(P('repeticiones')))
        self.csv = P('csv')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.laser_tf = None
        self.scan = None
        self.v = self.w = 0.0
        self.cmd_nav = (0.0, 0.0)
        self.t_cmd_nav = -1e9
        self.goal_handle = None
        self.filas_escritas = 0
        self.t_listo0 = None
        self.meta_actual = None
        self.reintentos = 0
        self.reenviar_en = None

        self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odom', self.on_odom, 10)
        self.create_subscription(Twist, P('cmd_in'), self.on_cmd, 10)
        self.pub = self.create_publisher(Twist, P('cmd_out'), 10)
        self.pub_estado = self.create_publisher(String, 'rl/estado', 10)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.create_timer(0.05, self.paso)
        self.t_camina0 = None
        self.set_estado = None
        self.set_pendiente = {}
        if self.caminantes:
            if SetEntityState is None:
                raise SystemExit('Hay caminantes pero falta gazebo_msgs (¿ROS con gazebo_ros?)')
            self.set_estado = [self.create_client(SetEntityState, nombre) for nombre in
                               ('/gazebo/set_entity_state', '/set_entity_state')]
            self.create_timer(0.05, self.mover_caminantes)
        self.create_timer(1.0, self.informe)
        self.fin_t = None

    def on_odom(self, msg):
        self.v, self.w = msg.twist.twist.linear.x, msg.twist.twist.angular.z

    def on_cmd(self, msg):
        self.cmd_nav = (msg.linear.x, msg.angular.z)
        self.t_cmd_nav = self.get_clock().now().nanoseconds * 1e-9

    def on_scan(self, msg):
        self.scan = msg
        if self.laser_tf is None:
            try:
                t = self.tf_buffer.lookup_transform(self.base, msg.header.frame_id, Time(),
                                                    timeout=Duration(seconds=0.0))
            except Exception:
                return
            q = t.transform.rotation
            self.laser_tf = (t.transform.translation.x, t.transform.translation.y, yaw_de(q))

    def pose_mapa(self):
        try:
            t = self.tf_buffer.lookup_transform(self.mapf, self.base, Time(),
                                                timeout=Duration(seconds=0.0))
        except Exception:
            return None
        return (t.transform.translation.x, t.transform.translation.y, yaw_de(t.transform.rotation))

    def scan_en_base(self):
        """(ángulos, rangos) respecto del lidar, en el frame de base_footprint."""
        m, (tx, ty, yaw) = self.scan, self.laser_tf
        r = np.asarray(m.ranges, float)
        a = m.angle_min + np.arange(r.size) * m.angle_increment + yaw
        ok = np.isfinite(r) & (r >= m.range_min)
        x = tx + r[ok] * np.cos(a[ok]) - C.LIDAR_X
        y = ty + r[ok] * np.sin(a[ok])
        rr = np.full(r.size, np.inf)
        rr[ok] = np.hypot(x, y)
        ang = np.zeros(r.size)
        ang[ok] = np.arctan2(y, x)
        return ang, rr

    def paso(self):
        out = Twist()
        pose = self.pose_mapa()
        if pose is not None and self.scan is not None and self.laser_tf is not None \
                and not self.mision.terminada:
            t = self.get_clock().now().nanoseconds * 1e-9
            if self.nav.server_is_ready() and self.esperar_arranque(t):
                ang, rr = self.scan_en_base()
                cmd = self.cmd_nav if t - self.t_cmd_nav < 0.5 else (0.0, 0.0)
                sal = self.mision.paso(t, pose, self.v, self.w, ang, rr, cmd,
                                       rango_max=self.scan.range_max)
                out.linear.x, out.angular.z = sal['v'], sal['w']
                if sal['cancelar'] and self.goal_handle is not None:
                    self.goal_handle.cancel_goal_async()
                if sal['meta'] is not None:
                    self.reintentos = 0
                    self.mandar_goal(sal['meta'])
                elif self.reenviar_en is not None and t >= self.reenviar_en:
                    self.reenviar_en = None
                    self.mandar_goal(self.meta_actual)
                self.guardar_filas()
        elif self.mision.terminada and self.fin_t is None:
            self.fin_t = self.get_clock().now()
            self.get_logger().info('Misión terminada.')
        self.pub.publish(out)

    def mover_caminantes(self):
        """Mueve a las personas en Gazebo (set_entity_state, 20 Hz) y le dice al filtro dónde
        están y a qué velocidad van (la verdad del simulador; el detector del adaptador sigue
        viéndolas solo cuando el lidar tiene un punto sobre ellas, con su ruido)."""
        cli = next((c for c in self.set_estado if c.service_is_ready()), None)
        t = self.get_clock().now().nanoseconds * 1e-9
        tau = 0.0 if self.t_camina0 is None else t - self.t_camina0
        for nombre, h, ruta in self.caminantes:
            pos, vel, yaw = ruta.estado(tau) if self.t_camina0 is not None else \
                (np.asarray(ruta.p[0], float), np.zeros(2), math.atan2(*(ruta.p[1] - ruta.p[0])[::-1]))
            h.pos, h.vel = np.asarray(pos, float), np.asarray(vel, float)
            if cli is None or self.set_pendiente.get(nombre) is not None \
                    and not self.set_pendiente[nombre].done():
                continue
            req = SetEntityState.Request()
            req.state.name = nombre
            req.state.reference_frame = 'world'
            req.state.pose.position.x, req.state.pose.position.y = float(pos[0]), float(pos[1])
            req.state.pose.orientation.z = math.sin(yaw / 2)
            req.state.pose.orientation.w = math.cos(yaw / 2)
            self.set_pendiente[nombre] = cli.call_async(req)

    def esperar_arranque(self, t):
        """True cuando pasaron ESPERA_ARRANQUE s con TF, scan y Nav2 listos (AMCL y los costmaps
        necesitan unos segundos; si la misión arranca antes, Nav2 rechaza la meta)."""
        if self.t_listo0 is None:
            self.t_listo0 = t
            self.get_logger().info(f'Nav2 listo; espero {ESPERA_ARRANQUE:.0f} s a que AMCL y los '
                                   'costmaps se asienten...')
        listo = t - self.t_listo0 >= ESPERA_ARRANQUE
        if listo and self.t_camina0 is None:
            self.t_camina0 = t
        return listo

    def mandar_goal(self, meta):
        self.meta_actual = meta
        g = NavigateToPose.Goal()
        g.pose = PoseStamped()
        g.pose.header.frame_id = self.mapf
        g.pose.header.stamp = self.get_clock().now().to_msg()
        g.pose.pose.position.x, g.pose.pose.position.y = float(meta[0]), float(meta[1])
        g.pose.pose.orientation.z = math.sin(meta[2] / 2)
        g.pose.pose.orientation.w = math.cos(meta[2] / 2)
        self.get_logger().info(f'Nav2 -> pre-agarre ({meta[0]:.2f}, {meta[1]:.2f})')
        fut = self.nav.send_goal_async(g)
        fut.add_done_callback(self.on_goal)

    def on_goal(self, fut):
        gh = fut.result()
        if not gh.accepted:
            self.reintentos += 1
            if self.reintentos <= MAX_REINTENTOS:
                self.get_logger().warn(f'Nav2 rechazó la meta (intento {self.reintentos}/'
                                       f'{MAX_REINTENTOS}); reintento en {T_REINTENTO:.0f} s')
                self.reenviar_en = self.get_clock().now().nanoseconds * 1e-9 + T_REINTENTO
            else:
                self.mision.nav_resultado(False)
            return
        self.reintentos = 0
        self.goal_handle = gh
        gh.get_result_async().add_done_callback(lambda f, gh=gh: self.on_result(f, gh))

    def on_result(self, fut, gh):
        ok = fut.result().status == 4
        if gh is self.goal_handle and self.mision.estado == 'navegando':
            self.mision.nav_resultado(ok)

    def guardar_filas(self):
        if not self.csv or self.filas_escritas == len(self.mision.filas):
            return
        nuevo = not os.path.exists(self.csv)
        os.makedirs(os.path.dirname(os.path.abspath(self.csv)), exist_ok=True)
        with open(self.csv, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNAS)
            if nuevo:
                w.writeheader()
            for f in self.mision.filas[self.filas_escritas:]:
                w.writerow(f)
                self.get_logger().info('Resultado: ' + ', '.join(f'{k}={v}' for k, v in f.items()))
        self.filas_escritas = len(self.mision.filas)

    def informe(self):
        th = self.ad.th
        self.pub_estado.publish(String(data=(
            f'{self.mision.estado} fase={self.ad.fase} margen={self.ad.margen():.2f} '
            f'd_stop_f={th.d_stop_f:.2f} d_stop_s={th.d_stop_s:.2f} '
            f'v_max={th.v_max:.2f} d_frente={self.ad.info_f["d_frente"]:.2f}')))
        if self.mision.terminada and self.fin_t is not None:
            self.guardar_filas()
            if (self.get_clock().now() - self.fin_t).nanoseconds * 1e-9 > 2.0:
                raise SystemExit(0)


def main():
    rclpy.init()
    nodo = NodoRL()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        nodo.pub.publish(Twist())
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
