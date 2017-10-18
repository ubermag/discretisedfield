FROM ubuntu:16.04

RUN apt update -y
RUN apt install -y git python3-pip curl
RUN python3 -m pip install --upgrade pip pytest-cov nbval \
      matplotlib pyvtk git+git://github.com/joommf/joommfutil.git

WORKDIR /usr/local
#RUN git clone https://github.com/joommf/discretisedfield.git
#RUN mkdir -p /usr/local/discretisedfield
# guessing here cloned repository might be ...
COPY ../discretisedfield/ /usr/local/discretisedfield/
#COPY * /usr/local/discretisedfield/.
WORKDIR /usr/local/discretisedfield
RUN python3 -m pip install .
