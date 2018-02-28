FROM ubuntu:16.04

MAINTAINER Sam Hardy

#python 3.6 repo
RUN apt-get update
RUN apt-get install -y software-properties-common vim
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y git

# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

#python specific dependencies
RUN pip3 install --upgrade scipy \
&& pip3 install numpy \
&& pip3 install tensorflow \
&& pip3 install keras \
&& pip3 install Pillow \
&& pip3 install h5py

#configure app directories
ADD /app /app
WORKDIR /app