"""Filtro de seguridad: cmd_vel_in (default /cmd_vel_teleop) + /scan -> cmd_vel_out.

Modo teleop (default): config:=safety.yaml, entrada /cmd_vel_teleop.
Modo Nav2: config:=safety_nav.yaml, entrada /safety/cmd_vel_in (sale de twist_mux).

Simulación: cmd_vel_out=/cmd_vel (plugin de Gazebo).
Milo físico: cmd_vel_out=/diff_drive_controller/cmd_vel_unstamped.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('cmd_vel_out', default_value='/cmd_vel'),
        DeclareLaunchArgument('cmd_vel_in', default_value='/cmd_vel_teleop'),
        DeclareLaunchArgument('config', default_value='safety.yaml'),
        Node(
            package='andesrobot_safety',
            executable='safety_filter',
            name='safety_filter',
            output='screen',
            parameters=[PathJoinSubstitution([FindPackageShare('andesrobot_safety'),
                                              'config', LaunchConfiguration('config')]),
                        {'use_sim_time': LaunchConfiguration('use_sim_time')}],
            remappings=[('cmd_vel', LaunchConfiguration('cmd_vel_out')),
                        ('cmd_vel_teleop', LaunchConfiguration('cmd_vel_in'))],
        ),
    ])
