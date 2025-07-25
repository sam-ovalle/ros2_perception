from setuptools import setup

package_name = 'advanced_perception'

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
            'yolo_object_detection = advanced_perception.yolo_object_detection:main',
            'yolo_person_follower = advanced_perception.yolo_person_follower:main',
            'yolo_pose_estimation = advanced_perception.yolo_pose_estimation:main',
            'yolo_segmentation = advanced_perception.yolo_segmentation:main',
        ],
    },
)
