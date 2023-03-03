from distutils.core import setup

setup(
    name='demo3_resistor',
    version='1.0.0',
    url='',
    license='BSD',
    author='Matthias De Ryck',
    author_email='matthias.deryck@kuleuven.be',
    description='Software for demo 3: obtaining the value of a resistor',
    packages=['demo3_resistor'],
    package_data={'demo3_resistor': ['data/*.npy']},
    package_dir={'': 'src'},
    scripts=['scripts/demo3_resistor.py'],
    install_requires=['opencv-python', 'numpy', 'matplotlib']
)