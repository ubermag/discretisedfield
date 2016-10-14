FROM ubuntu:16.04

RUN apt-get update -y
RUN apt-get install -y git python3-pip curl python3-tk
RUN python3 -m pip install --upgrade pip pytest-cov \
      matplotlib tornado ipython ipykernel \
      git+git://github.com/joommf/joommfutil.git \
      git+git://github.com/computationalmodelling/nbval.git nbformat

ENV MPLBACKEND Agg

WORKDIR /usr/local/

RUN git clone https://github.com/joommf/discretisedfield.git