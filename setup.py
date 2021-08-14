import json
import setuptools

# Extract metadata
with open('README.md', 'r') as f:
    long_description = f.read()
    
with open('requirements.txt', 'r') as f:
	install_requires = f.readlines()
 
with open('metadata.json', 'r') as f:
    data = json.load(f.read())

setuptools.setup(
    name=data['package'],
    version=data['version'],
    description=data['description'],
    author=', '.join(authors),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://ubermag.github.io',
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=install_requires,
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
