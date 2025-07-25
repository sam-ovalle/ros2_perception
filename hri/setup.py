from setuptools import setup
from glob import glob

package_name = 'hri'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include the data directory
        (f'share/{package_name}/data', glob('data/*.*')),
        # Include the data/person1 directory
        (f'share/{package_name}/data/person1', glob('data/person1/*.*')),
        # Include the data/mona_lisa directory
        (f'share/{package_name}/data/mona_lisa', glob('data/mona_lisa/*.*')),
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
            'face_detection = hri.face_detection:main',
            'eye_detection = hri.eye_detection:main',
            'face_recognition = hri.face_recognition:main',
            'human_tracking = hri.human_tracking:main',
        ],
    },
)
