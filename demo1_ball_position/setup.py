from distutils.core import setup

setup(
    name='demo1_ball_position',
    version='1.0.0',
    url='',
    license='BSD',
    author='Matthias De Ryck',
    author_email='matthias.deryck@kuleuven.be',
    description='Software for demo 1: obtaining the 3D position of a ball',
    packages=['demo1_ball_position'],
    package_data={'demo1_ball_position': ['data/*.npy']},
    package_dir={'': 'src'},
    scripts=['scripts/main.py'],
    install_requires=['pyrealsense2', 'opencv-python', 'numpy', 'matplotlib']
)