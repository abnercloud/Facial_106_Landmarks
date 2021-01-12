FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel

LABEL maintainer="wduo <wangduo@datatang.com>"


# Install necessary libraries.
COPY ./tsinghua_source/ /tsinghua_source/
RUN mv /tsinghua_source/etc/apt/sources.list /etc/apt/ \
  && mkdir /root/.pip/ && mv /tsinghua_source/root/.pip/pip.conf /root/.pip/

RUN apt-get -y update \
  && apt-get install -y libglib2.0-dev libsm6 libxext6 libxrender1

RUN pip install cython opencv-python==4.1.2.30 matplotlib==3.1.2 h5py tqdm \
  && pip install pycocotools==2.0.0 tensorboardX==1.9 visdom \
  && pip install tensorflow-gpu==1.8.0 mtcnn==0.0.9


# Add cuda9.0-cudnn7.0.5 for tensorflow-gpu==1.8.0, and update env variable.
COPY ./cuda9.0-cudnn7.0.5/ /cuda9.0-cudnn7.0.5/
RUN mv /cuda9.0-cudnn7.0.5/* /usr/local/cuda-9.0/lib64/ \
  && export CUDA_HOME=/usr/local/cuda-9.0/ \
  && export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}} \
  && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64" 


# Disable cudnn for batch_norm.
# ARG PYTORCH=/opt/conda/pkgs/pytorch-0.4.0-py36_cuda9.0.176_cudnn7.1.2_1/lib/python3.6/site-packages/torch
ARG PYTORCH=/opt/conda/lib/python3.6/site-packages/torch
RUN sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/nn/functional.py



