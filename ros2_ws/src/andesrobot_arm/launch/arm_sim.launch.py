"""Brazo de Milo en Gazebo: robot con el brazo controlado + IK + marcador interactivo + RViz.

Orden de lo que pasa:
 1. Se procesa el xacro de Milo con lock_arm:=false y arm_control:=true (brazo, lift y pinza
    con ros2_control) y gripper_camera:=true (cámara Orbbec sobre la pinza), y se publica en
    /robot_description.
 2. Gazebo abre el mundo y spawn_entity.py pone a Milo y la mesa de prueba frente a él.
 3. Si Milo apareció bien, el spawner arranca los controladores del brazo.
 4. arm_ik_node espera poses en /arm_target_pose; arm_marker_node las publica desde RViz.

Uso:  ros2 launch andesrobot_arm arm_sim.launch.py   [gui:=false] [rviz:=false]
                                                    [camara:=false] [mesa:=false]
Cámara: topics /gripper_camera/{color,depth,ir}/image_raw y /gripper_camera/depth/points
(ver andesrobot_description/urdf/andesrobot.gripper_camera.xacro). Guardar una imagen de cada
canal:  ros2 run andesrobot_arm capturar_camara
"""
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _strip_comments(node):
    """Borrar los comentarios XML del URDF (ver robot_description_urdf)."""
    for child in list(node.childNodes):
        if child.nodeType == child.COMMENT_NODE:
            node.removeChild(child)
        else:
            _strip_comments(child)


def robot_description_urdf(simple_collision, camara):
    xacro_file = os.path.join(get_package_share_directory('andesrobot_description'),
                              'urdf', 'andesrobot.urdf.xacro')
    controllers = os.path.join(get_package_share_directory('andesrobot_arm'),
                               'config', 'arm_controllers.yaml')
    doc = xacro.process_file(xacro_file, mappings={
        'lock_arm': 'false',
        'arm_control': 'true',
        'arm_controllers_file': controllers,
        'simple_collision': simple_collision,
        'gripper_camera': camara,
    })
    _strip_comments(doc)
    return doc.toxml()


def launch_setup(context):
    urdf = robot_description_urdf(LaunchConfiguration('simple_collision').perform(context),
                                  LaunchConfiguration('camara').perform(context))

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf, 'use_sim_time': True}],
    )

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'andesrobot', '-z', '0.02'],
        output='screen',
    )

    spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', 'arm_controller', 'gripper_controller',
                   '--controller-manager', '/controller_manager'],
        output='screen',
    )

    def start_controllers_if_spawned(event, context):
        if event.returncode != 0:
            return [LogInfo(msg='spawn_entity falló: no arranco los controladores. Si el '
                                "error es \"No module named 'rclpy._rclpy_pybind11'\", hay un "
                                'python3 que no es el de ROS primero en el PATH (Anaconda): '
                                'conda deactivate.')]
        return [spawner]

    controllers = RegisterEventHandler(
        OnProcessExit(target_action=spawn, on_exit=start_controllers_if_spawned))

    sim_time = [{'use_sim_time': True}]
    ik = Node(package='andesrobot_arm', executable='arm_ik_node',
              parameters=sim_time, output='screen')
    marker = Node(package='andesrobot_arm', executable='arm_marker_node',
                  parameters=sim_time, output='screen')
    rviz = Node(
        package='rviz2', executable='rviz2', output='screen',
        arguments=['-d', os.path.join(get_package_share_directory('andesrobot_arm'),
                                      'rviz', 'arm.rviz')],
        parameters=sim_time,
        condition=IfCondition(LaunchConfiguration('rviz')),
    )
    mesa = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-file', os.path.join(get_package_share_directory('andesrobot_arm'),
                                         'worlds', 'mesa_prueba.sdf'),
                   '-entity', 'mesa_prueba'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('mesa')),
    )
    return [rsp, spawn, controllers, ik, marker, rviz, mesa]


def generate_launch_description():
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'])),
        launch_arguments={'world': LaunchConfiguration('world'),
                          'gui': LaunchConfiguration('gui'),
                          'verbose': 'false'}.items(),
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=PathJoinSubstitution(
                [FindPackageShare('andesrobot_gazebo'), 'worlds', 'andesrobot_arena.world']),
            description='Ruta al .world'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='Abrir la ventana de Gazebo'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Abrir RViz con el marcador'),
        DeclareLaunchArgument('simple_collision', default_value='false',
                              description='true = colisiones con cajas en vez de mallas CAD'),
        DeclareLaunchArgument('camara', default_value='true',
                              description='Cámara Orbbec Gemini Plus sobre la pinza'),
        DeclareLaunchArgument('mesa', default_value='true',
                              description='Mesa con objetos de prueba frente a Milo'),
        gazebo,
        OpaqueFunction(function=launch_setup),
    ])
