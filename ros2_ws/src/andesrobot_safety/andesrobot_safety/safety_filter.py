"""Filtro de seguridad: /cmd_vel_teleop + /scan -> /cmd_vel.

Igual para simulación y para Milo físico. Frena si hay obstáculos a menos de
`stop_distance` en la dirección de avance; deja retroceder o girar para salir.

Este archivo es la parte "ROS" (topics, parámetros, timers). La matemática está en logic.py.

Entradas:  /scan (sensor_msgs/LaserScan)    lidar
           /cmd_vel_teleop (geometry_msgs/Twist)  lo que pides con el teclado
Salidas:   /cmd_vel (Twist)                 lo que realmente se le manda al robot
           /safety/state (String)           texto con lo que está haciendo el filtro
           /safety/stop_zone (PolygonStamped)  rectángulo para RViz
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point32, PolygonStamped, Twist
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from andesrobot_safety.logic import SafetyParams, filter_cmd, remove_self


class SafetyFilter(Node):

    def __init__(self):
        super().__init__('safety_filter')
        d = SafetyParams()
        for name, default in vars(d).items():
            self.declare_parameter(name, default)
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('scan_timeout', 0.5)
        self.declare_parameter('cmd_timeout', 0.0)

        self.p = SafetyParams(**{k: self.get_parameter(k).value for k in vars(d)})
        self.base_frame = self.get_parameter('base_frame').value
        self.scan_timeout = self.get_parameter('scan_timeout').value
        self.cmd_timeout = self.get_parameter('cmd_timeout').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.laser_tf = None

        self.points = np.zeros((0, 2))
        self.last_scan = None
        self.cmd = Twist()
        self.last_cmd = None
        self.last_state = ''

        self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Twist, 'cmd_vel_teleop', self.on_cmd, 10)
        self.pub_cmd = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pub_state = self.create_publisher(String, 'safety/state', 10)
        self.pub_zone = self.create_publisher(PolygonStamped, 'safety/stop_zone', 1)
        self.create_timer(1.0 / self.get_parameter('rate').value, self.step)
        self.create_timer(1.0, self.publish_zone)
        self.get_logger().info(
            'Filtro de seguridad: stop %.2f m, slow %.2f m, v_max %.2f m/s'
            % (self.p.stop_distance, self.p.slow_distance, self.p.max_linear))

    def on_cmd(self, msg):
        self.cmd = msg
        self.last_cmd = self.get_clock().now()

    def on_scan(self, msg):
        if self.laser_tf is None:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.base_frame, msg.header.frame_id, Time(), timeout=Duration(seconds=0.0))
            except Exception:
                return
            q = t.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.laser_tf = (t.transform.translation.x, t.transform.translation.y, yaw)
        r = np.asarray(msg.ranges, dtype=float)
        a = msg.angle_min + np.arange(r.size) * msg.angle_increment
        ok = np.isfinite(r) & (r >= max(msg.range_min, 0.05)) & (r <= msg.range_max)
        tx, ty, yaw = self.laser_tf
        ang = a[ok] + yaw
        pts = np.column_stack((tx + r[ok] * np.cos(ang), ty + r[ok] * np.sin(ang)))
        self.points = remove_self(pts, self.p)
        self.last_scan = self.get_clock().now()

    def step(self):
        now = self.get_clock().now()
        out = Twist()
        if self.last_scan is None or (now - self.last_scan).nanoseconds * 1e-9 > self.scan_timeout:
            state = 'PARADO: sin lidar'
        elif (self.cmd_timeout > 0.0 and self.last_cmd is not None
              and (now - self.last_cmd).nanoseconds * 1e-9 > self.cmd_timeout):
            state = 'sin comando'
        else:
            v, w, state = filter_cmd(self.cmd.linear.x, self.cmd.angular.z, self.points, self.p)
            out.linear.x, out.angular.z = v, w
        self.pub_cmd.publish(out)
        if state != self.last_state:
            self.pub_state.publish(String(data=state))
            if state.startswith('PARADO') or 'bloqueado' in state:
                self.get_logger().warn(state)
            self.last_state = state

    def publish_zone(self):
        p = self.p
        hw = p.footprint_half_width + p.side_margin
        xf, xr = p.footprint_x_max + p.stop_distance, p.footprint_x_min - p.stop_distance
        msg = PolygonStamped()
        msg.header.frame_id = self.base_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.polygon.points = [Point32(x=x, y=y, z=0.02) for x, y in
                              [(xr, -hw), (xf, -hw), (xf, hw), (xr, hw)]]
        self.pub_zone.publish(msg)


def main():
    rclpy.init()
    node = SafetyFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_cmd.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
