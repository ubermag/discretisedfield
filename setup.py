from distutils.core import setup

with open('README.rst') as f:
    readme = f.read()

setup(
    name='discretisedfield',
    version='0.1',
    description='A Python package for analysing and manipulating finite difference vector fields',
    long_description=readme,
    author='Computational Modelling Group',
    author_email='fangohr@soton.ac.uk',
    url = 'https://github.com/joommf/discretisedfield',
    #download_url = 'https://github.com/joommf/discretisedfield/tarball/0.1',
    packages=['discretisedfield',
              'discretisedfield.util',
              'discretisedfield.tests'],
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ]
)
