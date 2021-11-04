FROM tensorflow/tensorflow:2.6.0 as base-tf

ENV TZ=US
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-opencv

COPY requirements.txt /tmp/
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /tmp/requirements.txt
RUN python3 -m pip install pytest

COPY . /pytest-example/
WORKDIR /pytest-example/