# discretisedfield
Marijan Beg<sup>1,2</sup>, Ryan A. Pepper<sup>2</sup>, Thomas Kluyver<sup>1</sup>, and Hans Fangohr<sup>1,2</sup>

<sup>1</sup> European XFEL GmbH, Holzkoppel 4, 22869 Schenefeld, Germany  
<sup>2</sup> Faculty of Engineering and the Environment, University of Southampton, Southampton SO17 1BJ, United Kingdom  

| Description | Badge |
| --- | --- |
| Latest release | [![PyPI version](https://badge.fury.io/py/discretisedfield.svg)](https://badge.fury.io/py/discretisedfield) |
|                | [![Anaconda-Server Badge](https://anaconda.org/conda-forge/discretisedfield/badges/version.svg)](https://anaconda.org/conda-forge/discretisedfield) |
| License | [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) |
| Build | [![Build Status](https://travis-ci.org/joommf/discretisedfield.svg?branch=master)](https://travis-ci.org/joommf/discretisedfield) |
|       |  [![Build status](https://ci.appveyor.com/api/projects/status/83tcspfx3qlx6rlp/branch/master?svg=true)](https://ci.appveyor.com/project/marijanbeg/discretisedfield/branch/master) |
| Coverage | [![codecov](https://codecov.io/gh/joommf/discretisedfield/branch/master/graph/badge.svg)](https://codecov.io/gh/joommf/discretisedfield) |
| Documentation | [![Documentation Status](https://readthedocs.org/projects/discretisedfield/badge/?version=latest)](http://discretisedfield.readthedocs.io/en/latest/?badge=latest) |

## About

`discretisedfield` is a Python package that provides:

- Creating finite difference meshes

- Creating, analysis, and plotting of finite difference fields

- Reading and writing of different file types, such as `.ovf` and `.vtk`

- Available on all major operating systems (Windows, MacOS, Linux)

- Requires Python 3.5+

## Installation

`discretisedfield` can be installed using either `pip` or `conda` package managers.

### Python requirements

Before installing `discretisedfield` via `pip`, please make sure you have Python 3.5+ on your system. You can check that by running

    python3 --version

If you are using Linux, most probably you already have Python. However, on MacOS and Windows, this is usually not the case. If you do not have Python 3.5+ on your machine, we strongly recommend installing [Anaconda](https://www.anaconda.com/) Python distribution. [Download Anaconda](https://www.anaconda.com/download) for your operating system and follow instructions on the download page. Further information about installing Anaconda can be found [here](https://conda.io/docs/user-guide/install/download.html).

### `pip`

After installing Anaconda on MacOS or Windows, `pip` will also be installed. However, on Linux, if you do not have `pip`, you can install it with

    sudo apt install python3-pip

To install the `discretisedfield` version currently in the Python Package Index repository [PyPI](https://pypi.org/) run

    python3 -m pip install discretisedfield

#### Updating

`discretisedfield` can be updated to the latest released version by running

    python3 -m pip install --upgrade discretisedfield

However, if you want to use the most recent version of `discretisedfield` (still under development), you can do it by running

    git clone https://github.com/joommf/discretisedfield
    python3 -m pip install discretisedfield

**Note**: If you do not have `git` on your system, it can be installed by following these [instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

### `conda`

`discretisedfield` is installed using `conda` by running

    conda install --channel conda-forge discretisedfield

For further information on the `conda` package, dependency, and environment management, please have a look at its [documentation](https://conda.io/docs/). 

#### Updating

To update `discretisedfield` run

    conda upgrade discretisedfield

## Documentation

Documentation for `discretisedfield` is available [here](http://discretisedfield.readthedocs.io/en/latest/?badge=latest). APIs and tutorials (in the form of Jupyter notebooks) are available.

## Support

If you require support on installation or usage of `discretisedfield` or if you want to report a problem, you are welcome to raise an issue in our [joommf/help](https://github.com/joommf/help) repository.

## License

Licensed under the BSD 3-Clause "New" or "Revised" License. For details, please refer to the [LICENSE](LICENSE) file.

## How to cite

If you use `discretisedfield` in your research, please cite it as:

1. M. Beg, R. A. Pepper, and H. Fangohr. User interfaces for computational science: A domain specific language for OOMMF embedded in Python. [AIP Advances, 7, 56025](http://aip.scitation.org/doi/10.1063/1.4977225) (2017).

2. DOI will be available soon

## Acknowledgements

`discretisedfield` was developed as a part of [OpenDreamKit](http://opendreamkit.org/) – Horizon 2020 European Research Infrastructure project (676541).
