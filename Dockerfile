# syntax = docker/dockerfile:experimental
FROM ubuntu:20.04

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y curl wget ffmpeg libsm6 libxext6 zlib1g-dev python3.8 libpython3.8 python3.8-distutils

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.8 get-pip.py "pip==21.3.1" "setuptools==62.6.0"

# Other Python dependencies
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3.8 install -r requirements.txt

# Rest of the app
COPY . ./

EXPOSE 4449

CMD python3.8 -u dsp-server.py
