FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

MAINTAINER Jose Luis <j209820@dac.unicamp.br>

#
# Install Miniconda in /opt/conda
#

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
Run apt-get update

RUN mkdir /workspace/ && cd /workspace/ && git clone https://github.com/jose13579/variable-hyperparameter-image-impainting.git && cd variable-hyperparameter-image-impainting/ && conda env create -f environment.yml
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate vhii" >> ~/.bashrc

RUN apt-get install libxrender1

WORKDIR /workspace
