"""Navegación AUTÓNOMA en simulación: Gazebo + Milo + filtro lidar + AMCL + Nav2 + RViz.

Lo corre:   ./sim.sh nav <mapa> [casa|arena]
Mandar a Milo:  ./sim.sh ir kitchen      (o "2D Goal Pose" en RViz)
Teleop encima de Nav2 (le quita el control mientras tocas teclas):  ./sim.sh teleop

Flujo de datos:
  Gazebo -> /scan -> filtro -> /scan_filtered -> AMCL (dónde estoy en el mapa) + costmaps
  Nav2 -> /nav/cmd_vel --\
  teclado -> /cmd_vel_teleop --> twist_mux -> /safety/cmd_vel_in -> safety_filter
                                                  -> /cmd_vel -> Gazebo

OJO: en Gazebo la odometría sale de la pose REAL del robot en el mundo (diff drive con
odometry_source = world), así que el frame "map" del SLAM queda igual al mundo de Gazebo.
Por eso AMCL parte en la misma pose donde aparece Milo (x/y/yaw, los pasa ./sim.sh).
"""

import math

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler,
                            TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_gazebo = FindPackageShare('andesrobot_gazebo')
    pkg_slam = FindPackageShare('andesrobot_slam')
    pkg_safety = FindPackageShare('andesrobot_safety')
    pkg_nav = FindPackageShare('andesrobot_navigation')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_gazebo, 'launch', 'sim.launch.py'])
        ),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'gui': LaunchConfiguration('gazebo_gui'),
            'x': LaunchConfiguration('x'),
            'y': LaunchConfiguration('y'),
            'yaw': LaunchConfiguration('yaw'),
        }.items(),
    )
    scan_filter = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_slam, 'launch', 'mapping.launch.py'])
        ),
        launch_arguments={'use_sim_time': 'true', 'slam': 'false'}.items(),
    )
    safety = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_safety, 'launch', 'safety.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'config': 'safety_nav.yaml',
            'cmd_vel_in': '/safety/cmd_vel_in',
            'cmd_vel_out': '/cmd_vel',
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('filtro'), "' == 'safety'"])),
    )
    rl = ExecuteProcess(
        cmd=['python3', '/rl/ros/nodo_rl.py', '--ros-args',
             '-p', 'use_sim_time:=true',
             '-p', ['modo:=', LaunchConfiguration('modo')],
             '-p', ['politica:=', LaunchConfiguration('politica')],
             '-p', ['objetos:=', LaunchConfiguration('objetos')],
             '-p', ['repeticiones:=', LaunchConfiguration('repeticiones')],
             '-p', ['csv:=', LaunchConfiguration('csv')],
             '-p', ['personas_yaml:=', LaunchConfiguration('personas_yaml')],
             '-p', 'objetos_yaml:=/ros2_ws/src/andesrobot_gazebo/worlds/'
                   'robocup_casa_objetos.yaml'],
        additional_env={'PYTHONDONTWRITEBYTECODE': '1'},
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('filtro'), "' == 'rl'"])),
    )
    rl_fin = RegisterEventHandler(
        OnProcessExit(target_action=rl, on_exit=[EmitEvent(event=Shutdown())]))
    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'navigation.launch.py'])
        ),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'use_sim_time': 'true',
            'ubicaciones': LaunchConfiguration('ubicaciones'),
            'x': LaunchConfiguration('x'),
            'y': LaunchConfiguration('y'),
            'yaw': LaunchConfiguration('yaw'),
            'amcl_tf': PythonExpression(
                ["'true' if '", LaunchConfiguration('localizacion'), "' == 'amcl' else 'false'"]),
        }.items(),
    )
    mapa_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'map', '--child-frame-id', 'odom'],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('localizacion'), "' == 'mundo'"])),
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([pkg_nav, 'rviz', 'navigation.rviz'])],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    def _pose_inicial(context):
        x = float(LaunchConfiguration('x').perform(context))
        y = float(LaunchConfiguration('y').perform(context))
        yaw = float(LaunchConfiguration('yaw').perform(context))
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.1
        cov[35] = 0.05
        msg = ('{header: {frame_id: map}, pose: {pose: {position: {x: %f, y: %f, z: 0.0}, '
               'orientation: {x: 0.0, y: 0.0, z: %f, w: %f}}, covariance: %s}}'
               % (x, y, math.sin(yaw / 2), math.cos(yaw), str(cov)))
        return [ExecuteProcess(
            cmd=['ros2', 'topic', 'pub', '-t', '8', '-r', '1', '/initialpose',
                 'geometry_msgs/msg/PoseWithCovarianceStamped', msg],
            output='screen')]

    pose_inicial = TimerAction(period=20.0, actions=[OpaqueFunction(function=_pose_inicial)])

    return LaunchDescription(
        [
            DeclareLaunchArgument('map', description='Ruta al .yaml del mapa'),
            DeclareLaunchArgument('mundo', default_value='robocup_casa'),
            DeclareLaunchArgument(
                'world',
                default_value=PathJoinSubstitution(
                    [pkg_gazebo, 'worlds', [LaunchConfiguration('mundo'), '.world']]
                ),
            ),
            DeclareLaunchArgument(
                'ubicaciones',
                default_value='/ros2_ws/src/andesrobot_slam/maps/casa_lugares.yaml',
            ),
            DeclareLaunchArgument('x', default_value='0.0'),
            DeclareLaunchArgument('y', default_value='0.0'),
            DeclareLaunchArgument('yaw', default_value='0.0'),
            DeclareLaunchArgument('gazebo_gui', default_value='true'),
            DeclareLaunchArgument('filtro', default_value='safety'),
            DeclareLaunchArgument('modo', default_value='ppo'),
            DeclareLaunchArgument('politica', default_value='ninguna'),
            DeclareLaunchArgument('objetos', default_value='taza_comedor'),
            DeclareLaunchArgument('repeticiones', default_value='1'),
            DeclareLaunchArgument('csv', default_value='ninguna'),
            DeclareLaunchArgument('personas_yaml', default_value='ninguna'),
            DeclareLaunchArgument('rviz', default_value='true'),
            DeclareLaunchArgument('localizacion', default_value='mundo'),
            sim,
            scan_filter,
            safety,
            rl,
            rl_fin,
            nav,
            mapa_odom,
            rviz,
            pose_inicial,
        ]
    )
