import setuptools

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='discretisedfield',
    version='0.8.8',
    description=('Python package for definition, reading, '
                 'and visualisation of finite difference fields.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://ubermag.github.io',
    author='Marijan Beg, Ryan A. Pepper, Thomas Kluyver, and Hans Fangohr',
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "ovf2vtk = discretisedfield.ovf2vtk:main",
        ],
    },
    include_package_data=True,
    install_requires=['ubermagutil',
                      'matplotlib',
                      'jupyterlab',
                      'seaborn',
                      'h5py',
                      'pyvtk',
                      'k3d'],
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: BSD License',
                 'Programming Language :: Python :: 3 :: Only',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Operating System :: Microsoft :: Windows',
                 'Topic :: Scientific/Engineering :: Physics',
                 'Intended Audience :: Science/Research',
                 'Natural Language :: English']
)
