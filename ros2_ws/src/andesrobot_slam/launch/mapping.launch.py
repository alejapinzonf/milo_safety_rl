"""Mapeo 2D: filtro del RPLIDAR C1 (/scan -> /scan_filtered) + slam_toolbox online async.

Flujo de datos:
  lidar (/scan) --> laser_filters (saca la columna del robot) --> /scan_filtered
  /scan_filtered + odometría (TF odom->base_footprint) --> slam_toolbox --> /map + TF map->odom

SLAM = Simultaneous Localization And Mapping: arma el mapa y al mismo tiempo calcula dónde
está el robot dentro de ese mapa. "online async" = procesa los scans en vivo y, si llega uno
mientras procesa otro, se salta el nuevo (no se atrasa).
Este mismo launch se usa en simulación (use_sim_time:=true) y en Milo físico (false).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('andesrobot_slam')
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam_params = LaunchConfiguration('slam_params')

    scan_filter = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        name='scan_to_scan_filter_chain',
        parameters=[PathJoinSubstitution([pkg, 'config', 'rplidar_c1_filter.yaml']),
                    {'use_sim_time': use_sim_time}],
        remappings=[('scan', 'scan'), ('scan_filtered', 'scan_filtered')],
        output='screen',
    )

    slam = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[slam_params, {'use_sim_time': use_sim_time}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('slam')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('slam', default_value='true'),
        DeclareLaunchArgument(
            'slam_params',
            default_value=PathJoinSubstitution([pkg, 'config', 'slam_toolbox_mapping.yaml'])),
        scan_filter,
        slam,
    ])
