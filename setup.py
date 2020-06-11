import setuptools

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='discretisedfield',
    version='0.8.12',
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
    python_requires='>=3.6',
    install_requires=['ubermagutil==0.2.6',
                      'matplotlib>=3.2',
                      'pandas>=1.0',
                      'jupyterlab>=2.1',
                      'h5py>=2.10',
                      'k3d>=2.8'],
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Education',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Natural Language :: English',
                 'Operating System :: MacOS',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: Unix',
                 'Programming Language :: Python :: 3 :: Only',
                 'Topic :: Scientific/Engineering :: Physics',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 'Topic :: Scientific/Engineering :: Visualization']
)
