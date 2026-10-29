import os
from glob import glob

from setuptools import setup

package_name = 'andesrobot_slam'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.*')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*.*') + glob('maps/.gitkeep')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aleja',
    maintainer_email='m.pinzonfarfan@uandresbello.edu',
    description='Mapeo 2D con slam_toolbox + filtro del RPLIDAR C1 para AndesRobot.',
    license='Apache-2.0',
    entry_points={'console_scripts': []},
)
