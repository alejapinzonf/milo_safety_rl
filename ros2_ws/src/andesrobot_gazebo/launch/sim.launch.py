"""AndesRobot en Gazebo Classic 11: mundo + robot_state_publisher + spawn.

Expone: /scan, /cmd_vel, /odom, /imu, /joint_states, TF odom->base_footprint.

Orden de lo que pasa:
 1. Gazebo abre el mundo (la arena).
 2. robot_state_publisher publica el URDF en /robot_description (versión simulación).
 3. spawn_entity.py lee ese URDF y "pone" a Milo dentro de Gazebo.
 4. Los plugins del URDF (andesrobot.sim.xacro) empiezan a publicar lidar, odometría, IMU.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_gazebo = FindPackageShare('andesrobot_gazebo')
    pkg_desc = FindPackageShare('andesrobot_description')

    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')
    verbose = LaunchConfiguration('verbose')
    x, y, yaw = LaunchConfiguration('x'), LaunchConfiguration('y'), LaunchConfiguration('yaw')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'])),
        launch_arguments={'world': world, 'gui': gui, 'verbose': verbose}.items(),
    )

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([pkg_desc, 'launch', 'rsp.launch.py'])),
        launch_arguments={'use_sim_time': 'true',
                          'lock_arm': 'true',
                          'simple_collision': LaunchConfiguration('simple_collision'),
                          'sim_lidar': LaunchConfiguration('lidar'),
                          'sim_imu': LaunchConfiguration('imu')}.items(),
    )

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'andesrobot',
                   '-x', x, '-y', y, '-z', '0.02', '-Y', yaw],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=PathJoinSubstitution([pkg_gazebo, 'worlds', 'andesrobot_arena.world']),
            description='Ruta al .world'),
        DeclareLaunchArgument('gui', default_value='true', description='Abrir gzclient'),
        DeclareLaunchArgument('verbose', default_value='true',
                              description='Logs de Gazebo ([Err]/[Wrn]) en la terminal'),
        DeclareLaunchArgument('simple_collision', default_value='true',
                              description='false = usar mallas CAD como colisión (más pesado)'),
        DeclareLaunchArgument('lidar', default_value='true', description='Plugin RPLIDAR C1'),
        DeclareLaunchArgument('imu', default_value='true', description='Plugin IMU'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        gazebo,
        rsp,
        spawn,
    ])
