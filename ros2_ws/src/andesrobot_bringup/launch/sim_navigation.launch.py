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

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            IncludeLaunchDescription, RegisterEventHandler)
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

    # 1) Gazebo + mundo + Milo
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
    # 2) Solo el filtro del lidar (slam:=false: aquí localiza AMCL sobre el mapa guardado)
    scan_filter = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_slam, 'launch', 'mapping.launch.py'])
        ),
        launch_arguments={'use_sim_time': 'true', 'slam': 'false'}.items(),
    )
    # 3) Filtro de seguridad en modo navegación (lee lo que sale de twist_mux)
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
        # filtro:=rl lo reemplaza por el filtro + escudo del paper con el PPO (paso 3b)
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('filtro'), "' == 'safety'"])),
    )
    # 3b) Validación del paper (E5): nodo con el filtro adaptativo + escudo + PPO, en vez del
    # safety_filter. Lee lo mismo (/scan, /safety/cmd_vel_in) y publica en /cmd_vel. Además manda
    # a Nav2 al pre-agarre de cada objeto y mide. Detalle: rl/LEEME_GAZEBO.md
    # (rl/ está montado en el contenedor como /rl, ver docker/compose.yaml)
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
    # Cuando el nodo termina su misión, se apaga todo el launch (así se puede encadenar corridas).
    rl_fin = RegisterEventHandler(
        OnProcessExit(target_action=rl, on_exit=[EmitEvent(event=Shutdown())]))
    # 4) AMCL + Nav2 + twist_mux + marcadores de ubicaciones
    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'navigation.launch.py'])
        ),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'use_sim_time': 'true',
            'ubicaciones': LaunchConfiguration('ubicaciones'),
            # AMCL parte donde aparece Milo (map = mundo en la sim)
            'x': LaunchConfiguration('x'),
            'y': LaunchConfiguration('y'),
            'yaw': LaunchConfiguration('yaw'),
        }.items(),
    )
    # 5) RViz de navegación
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([pkg_nav, 'rviz', 'navigation.rviz'])],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

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
            # Dónde aparece Milo en el mundo (y dónde parte AMCL). ./sim.sh nav pasa los de la casa.
            DeclareLaunchArgument('x', default_value='0.0'),
            DeclareLaunchArgument('y', default_value='0.0'),
            DeclareLaunchArgument('yaw', default_value='0.0'),
            DeclareLaunchArgument('gazebo_gui', default_value='true'),
            # filtro: safety (el de siempre) | rl (filtro adaptativo + escudo + PPO, para el paper)
            DeclareLaunchArgument('filtro', default_value='safety'),
            DeclareLaunchArgument('modo', default_value='ppo'),  # ppo | nominal | minimo | reglas
            DeclareLaunchArgument('politica', default_value='ninguna'),  # .npz (solo modo ppo)
            DeclareLaunchArgument('objetos', default_value='taza_comedor'),  # lista con comas
            DeclareLaunchArgument('repeticiones', default_value='1'),
            DeclareLaunchArgument('csv', default_value='ninguna'),
            DeclareLaunchArgument('personas_yaml', default_value='ninguna'),
            DeclareLaunchArgument('rviz', default_value='true'),
            sim,
            scan_filter,
            safety,
            rl,
            rl_fin,
            nav,
            rviz,
        ]
    )
