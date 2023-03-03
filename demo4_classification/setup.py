from distutils.core import setup

setup(
    name='demo4_classification',
    version='1.0.0',
    url='',
    license='BSD',
    author='Matthias De Ryck',
    author_email='matthias.deryck@kuleuven.be',
    description='Software for demo 4: classification of defects on a flat sheet',
    packages=['demo4_classification'],
    package_data={'demo4_classification': ['data/*.npy']},
    package_dir={'': 'src'},
    scripts=['scripts/demo4_classification.py'],
    install_requires=['opencv-python', 'numpy', 'matplotlib']
)