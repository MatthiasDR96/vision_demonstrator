from distutils.core import setup

setup(
    name='demo3_resistors',
    version='1.0.0',
    url='',
    license='BSD',
    author='Matthias De Ryck',
    author_email='matthias.deryck@kuleuven.be',
    description='Software for demo 3: obtaining the value of a resistor',
    packages=['demo3_resistors'],
    package_data={'demo3_resistors': ['data/*.npy', 'data/*.csv', 'data/*.sav', 'data/images/*.jpg']},
    package_dir={'': 'src'},
    scripts=['scripts/main.py'],
    install_requires=['opencv-python', 'numpy', 'matplotlib']
)