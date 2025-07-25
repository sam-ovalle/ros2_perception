from setuptools import setup

package_name = 'bench_detector'

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
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'surface_detection = bench_detector.surface_detection:main',
            'static_transform_publisher = bench_detector.static_transform_publisher:main',
            'object_detection = bench_detector.object_detection:main',
        ],
    },
)
