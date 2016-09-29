FROM ubuntu:16.04

MAINTAINER Marijan Beg <m.beg@soton.ac.uk>

RUN apt-get update -y && \
    apt-get install -y git python3-pip curl && \
    python3 -m pip install --upgrade pip pytest-cov codecov nbval \
      matplotlib tornado ipython ipykernel 

# Set locale environment variables for nbval tests.
RUN locale-gen en_GB.UTF-8
ENV LANG en_GB.UTF-8
ENV LANGUAGE en_GB:en
ENV LC_ALL en_GB.UTF-8

WORKDIR /usr/local/

RUN git clone https://github.com/joommf/discretisedfield.git