from distutils.core import setup

setup(
    name='demo2_sawblade',
    version='1.0.0',
    url='',
    license='BSD',
    author='Matthias De Ryck',
    author_email='matthias.deryck@kuleuven.be',
    description='Software for demo 2: obtaining the tooth angle of a sawblade',
    packages=['demo2_sawblade'],
    package_data={'demo2_sawblade': ['data/*.npy']},
    package_dir={'': 'src'},
    scripts=['scripts/demo2_sawblade.py'],
    install_requires=['opencv-python', 'numpy', 'matplotlib']
)