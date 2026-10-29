import os
from glob import glob

from setuptools import setup

package_name = 'andesrobot_description'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*.*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.*')),
        (os.path.join('share', package_name, 'docs'), glob('docs/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aleja',
    maintainer_email='m.pinzonfarfan@uandresbello.edu',
    description='Modelo URDF/xacro del robot AndesRobot (base diferencial, columna elevadora, brazo 6DOF, RPLIDAR C1, IMU).',
    license='Apache-2.0',
    entry_points={'console_scripts': []},
)
