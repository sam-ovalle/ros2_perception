from setuptools import setup

package_name = 'line_following'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='samovalle878@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'follow_yellow_line = line_following.follow_yellow_line:main',
            'follow_yellow_line_optimized = line_following.follow_yellow_line_optimized:main',
            'follow_door = line_following.follow_door:main',
        ],
    },
)
