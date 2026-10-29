import os
from glob import glob

from setuptools import setup

package_name = 'andesrobot_safety'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aleja',
    maintainer_email='m.pinzonfarfan@uandresbello.edu',
    description='Filtro de seguridad de Milo.',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'safety_filter = andesrobot_safety.safety_filter:main',
    ]},
)
