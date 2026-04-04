from setuptools import find_packages, setup

package_name = 'aruco_3d_locator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jwg',
    maintainer_email='wjddnrud4487@kw.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'aruco_3d_node = aruco_3d_locator.aruco_3d_node:main',
            'aruco_3d_node_r = aruco_3d_locator.aruco_3d_node_r:main',
            'aruco_3d_node_zed = aruco_3d_locator.aruco_3d_node_zed:main',
        ],
    },
)
