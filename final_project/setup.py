from setuptools import find_packages, setup

package_name = 'final_project'

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
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'go_to_aoi = final_project.go_to_aoi:main',
            'inventory_checking = final_project.inventory_checking:main',
            'line_follower = final_project.line_follower:main',
            'object_detection = final_project.object_detection:main',
        ],
    },
)
