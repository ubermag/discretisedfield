import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='discretisedfield',
    version='0.8.15',
    description=('Python package for definition, reading, '
                 'and visualisation of finite-difference fields.'),
    author=('Marijan Beg, Martin Lang, Ryan A. Pepper, '
            'Thomas Kluyver, and Hans Fangohr'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://ubermag.github.io',
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=['ubermagutil==0.2.7',
                      'matplotlib>=3.3',
                      'pandas>=1.2',
                      'jupyterlab==2.2',
                      'h5py>=3.1',
                      'k3d>=2.9'],
    entry_points={
        "console_scripts": [
            "ovf2vtk = discretisedfield.ovf2vtk:main",
        ],
    },
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
