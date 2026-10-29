"""Visualiza AndesRobot en RViz con sliders para las articulaciones.

Uso:  ros2 launch andesrobot_description display.launch.py
Sirve para revisar el modelo (URDF) sin Gazebo: mueves cada articulación con un slider y
ves en RViz cómo se mueve el robot.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('andesrobot_description')
    gui = LaunchConfiguration('gui')

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([pkg, 'launch', 'rsp.launch.py'])),
        launch_arguments={'use_sim_time': 'false', 'lock_arm': 'false'}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true',
                              description='Sliders de joint_state_publisher_gui'),
        rsp,
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
             condition=IfCondition(gui)),
        Node(package='joint_state_publisher', executable='joint_state_publisher',
             condition=UnlessCondition(gui)),
        Node(package='rviz2', executable='rviz2', output='screen',
             arguments=['-d', PathJoinSubstitution([pkg, 'rviz', 'display.rviz'])]),
    ])
