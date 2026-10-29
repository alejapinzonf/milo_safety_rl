"""Navegación autónoma de Milo: localización (map_server + AMCL) + Nav2 + twist_mux.

Sirve igual para simulación (use_sim_time:=true) y para Milo físico (false).
NO levanta el robot, el filtro del lidar ni el filtro de seguridad: eso lo hacen
andesrobot_bringup/launch/sim_navigation.launch.py (y robot_navigation.launch.py).

Entradas:  /scan_filtered, /odom, TF odom->base_footprint, un mapa .yaml (map:=)
Salidas:   /safety/cmd_vel_in  (va al filtro de seguridad, que lo manda al robot)
           /map, TF map->odom, /plan, costmaps, acción /navigate_to_pose

Velocidades (por qué tantos remaps): ver el comentario al inicio de config/nav2_params.yaml.
Se escribe cada nodo a mano (en vez de incluir nav2_bringup) para controlar esos remaps:
en nav2_bringup (Humble) el behavior_server publica directo en /cmd_vel y se saltaría el mux.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('andesrobot_navigation')
    use_sim_time = LaunchConfiguration('use_sim_time')
    yaml_nav2 = os.path.join(get_package_share_directory('andesrobot_navigation'), 'config',
                             'nav2_params.yaml')
    params = ['--params-file', yaml_nav2]
    autostart = LaunchConfiguration('autostart')

    common = [{'use_sim_time': use_sim_time}]
    tf = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        ros_arguments=params,
        parameters=common + [{'yaml_filename': LaunchConfiguration('map')}],
        remappings=tf,
    )
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        ros_arguments=params,
        parameters=common
        + [
            {
                'initial_pose.x': ParameterValue(LaunchConfiguration('x'), value_type=float),
                'initial_pose.y': ParameterValue(LaunchConfiguration('y'), value_type=float),
                'initial_pose.yaw': ParameterValue(LaunchConfiguration('yaw'), value_type=float),
                'tf_broadcast': ParameterValue(LaunchConfiguration('amcl_tf'), value_type=bool),
            }
        ],
        remappings=tf,
    )
    lm_loc = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'autostart': autostart,
                'node_names': ['map_server', 'amcl'],
            }
        ],
    )

    controller = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        ros_arguments=params,
        parameters=common,
        remappings=tf + [('cmd_vel', '/nav/cmd_vel_raw')],
    )
    smoother = Node(
        package='nav2_smoother',
        executable='smoother_server',
        name='smoother_server',
        output='screen',
        ros_arguments=params,
        parameters=common,
        remappings=tf,
    )
    planner = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        ros_arguments=params,
        parameters=common,
        remappings=tf,
    )
    behaviors = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        ros_arguments=params,
        parameters=common,
        remappings=tf + [('cmd_vel', '/nav/cmd_vel')],
    )
    bt = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        ros_arguments=params,
        parameters=common,
        remappings=tf,
    )
    waypoints = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        ros_arguments=params,
        parameters=common,
        remappings=tf,
    )
    vel_smoother = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='screen',
        ros_arguments=params,
        parameters=common,
        remappings=tf + [('cmd_vel', '/nav/cmd_vel_raw'), ('cmd_vel_smoothed', '/nav/cmd_vel')],
    )
    lm_nav = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'autostart': autostart,
                'node_names': [
                    'controller_server',
                    'smoother_server',
                    'planner_server',
                    'behavior_server',
                    'bt_navigator',
                    'waypoint_follower',
                    'velocity_smoother',
                ],
            }
        ],
    )

    mux = Node(
        package='twist_mux',
        executable='twist_mux',
        name='twist_mux',
        output='screen',
        parameters=[
            PathJoinSubstitution([pkg, 'config', 'twist_mux.yaml']),
            {'use_sim_time': use_sim_time},
        ],
        remappings=[('cmd_vel_out', '/safety/cmd_vel_in')],
    )

    markers = Node(
        package='andesrobot_navigation',
        executable='marcadores',
        name='ubicaciones_rviz',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time, 'ubicaciones': LaunchConfiguration('ubicaciones')}
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('map', description='Ruta al .yaml del mapa (de ./sim.sh mapa)'),
            DeclareLaunchArgument('use_sim_time', default_value='true'),
            DeclareLaunchArgument('autostart', default_value='true'),
            DeclareLaunchArgument('amcl_tf', default_value='true'),
            DeclareLaunchArgument(
                'ubicaciones',
                default_value='/ros2_ws/src/andesrobot_slam/maps/casa_lugares.yaml',
            ),
            DeclareLaunchArgument('x', default_value='0.0'),
            DeclareLaunchArgument('y', default_value='0.0'),
            DeclareLaunchArgument('yaw', default_value='0.0'),
            map_server,
            amcl,
            lm_loc,
            controller,
            smoother,
            planner,
            behaviors,
            bt,
            waypoints,
            vel_smoother,
            lm_nav,
            mux,
            markers,
        ]
    )
