FROM ubuntu:16.04

RUN apt update -y
RUN apt install -y git python3-pip curl
RUN python3 -m pip install --upgrade pip

COPY . /usr/local/discretisedfield/

RUN python3 -m pip install /usr/local/discretisedfield

# check dependencies for tests run from the module are fulfilled.
# not sure this is good here, as the container building fails if
# the tests fail. Probably better if we had a dedicated container for this test
#
# Actual test is:
RUN python3 -c "import discretisedfield as d; d.test()"

# install additional libraries we only need for testing and documentation
RUN python3 -m pip install pytest-cov nbval

# CD into
WORKDIR /usr/local/discretisedfield
