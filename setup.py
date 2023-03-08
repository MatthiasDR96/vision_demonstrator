from distutils.core import setup

setup(
    name='vision_demonstrator',
    version='1.0.0',
    url='',
    license='BSD',
    author='Matthias De Ryck',
    author_email='matthias.deryck@kuleuven.be',
    description='Software for the vision demonstrator of KU Leuven Bruges',
    packages=['vision_demonstrator'],
    package_data={'vision_demonstrator': ['data/*.npy', 'data/*.nav', 'data/*.scv', 'config/*.yaml']},
    package_dir={'': 'src'},
    scripts=['scripts/main/demo1_main.py', 'scripts/main/demo2_main.py', 'scripts/main/demo3_main.py', 'scripts/main/demo4_main.py'],
    install_requires=['paho-mqtt', 'pyrealsense2', 'opencv-python', 'numpy', 'matplotlib', 'pandas']
)