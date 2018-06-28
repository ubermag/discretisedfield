# discretisedfield
Marijan Beg<sup>1,2</sup>, Ryan A. Pepper<sup>2</sup>, and Hans Fangohr<sup>1,2</sup>

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

This module is a Python package that provides:

- Creating finite difference meshes

- Creating and analysis of finite difference fields

- Plotting of finite difference field

- Reading and writing of different files, such as `.ovf` and `.vtk`

- Available on all major operating systems (Windows, MacOS, Linux)

- Available for Python 3.5+

## Installation and upgrade

This module (`discretisedfield`) can be installed using either `pip` or `conda` package managers.

### Requirements

Before running installing `discretisedfield` via `pip`, please make sure you have Python 3.5+ on your system. You can check that by running

    python3 --version

If you are using Linux, you probably already have Python installed. However, on MacOS and Windows, this is usually not the case. If you do not have Python 3.5+ on your machine, we stronly recommend installing [Anaconda](https://www.anaconda.com/) Python distribution. Firstly, [download](https://www.anaconda.com/download/#linux) Anaconda for your operating system and then follow instructions on the download page. Further information about installing Anaconda can be found [here](https://conda.io/docs/user-guide/install/download.html).

When you install Python using Anaconda distribution on MacOS or Windows, you will also install `pip`. However, on Linux, if you do not have it installed, you can get it with

    sudo apt install python3-pip

### `pip`

To install the `discretisedfield` version currently in [PyPI](https://pypi.org/) - the Python Package Index repository, run in your terminal

    python3 -m pip install discretisedfield

If you already have the package installed, you can upgrade it by running

    python3 -m pip install --upgrade discretisedfield

On the other had, if you want to use the most recent version of `discretisedfield` which is still under development, you can do it with

    git clone https://github.com/joommf/discretisedfield
    cd discretisedfield
    python3 -m pip install .

Note: If you do not have `git` on your system, please follow instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

### `conda`

You can install `discretisedfield` via `conda` by running

    conda install discretisedfield

To upgrade `discretisedfield` which you previously installed using `conda`, you can run

    conda upgrade discretisedfield

For further information on the `conda` package, dependency and environment management, please look at its documentation [here](https://conda.io/docs/). 

## Documentation

Documentation for this module is available [here](http://discretisedfield.readthedocs.io/en/latest/?badge=latest). Both APIs for all modules as well as the tutorials in the form of Jupyter notebooks are available.

## Support

If you require support on installation or usage of this module as well as if you want to report a problem, you are welcome to raise an issue in our [joommf/help](https://github.com/joommf/help) repository.

## How to cite

If you use this OOMMF extension in your research, please cite it as:

1. M. Beg, R. A. Pepper, and H. Fangohr. User interfaces for computational science: A domain specific language for OOMMF embedded in Python. [AIP Advances, 7, 56025](http://aip.scitation.org/doi/10.1063/1.4977225) (2017).

2. DOI will be available soon

## License

This extension is licensed under the BSD 3-Clause "New" or "Revised" License. For details, please refer to the [LICENSE](LICENSE) file.

## Acknowledgements

This extension was developed as a part of [OpenDreamKit](http://opendreamkit.org/) – Horizon 2020 European Research Infrastructure project (676541).
