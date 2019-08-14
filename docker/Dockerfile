FROM ubuntu:18.04

RUN apt update -y
RUN apt install -y git python3-pip curl
RUN python3 -m pip install --upgrade pip pytest-cov nbval \
      git+git://github.com/ubermag/ubermagutil.git

COPY . /usr/local/discretisedfield/
WORKDIR /usr/local/discretisedfield
RUN python3 -m pip install .
