# Base image
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# System packages 
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y apt-utils \
    && apt-get install -y curl \
    && /bin/echo -e "8\n37\n" | apt-get -y install git-all \
    && /bin/echo -e "8\n37\n" | apt-get -y install texlive-latex-extra

# Miniconda installation python 3.7 until opencv is available for higher versions of python
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
    && bash Miniconda3-py37_4.9.2-Linux-x86_64.sh -p /miniconda -b \
    && rm Miniconda3-py37_4.9.2-Linux-x86_64.sh

# RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b \
#     && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda \
    && conda init bash \
    && echo "cd code" >> ~\.bashrc \
    && echo "conda activate base" >> ~/.bashrc

# Install matplotlib and Tensorflow
RUN conda install tensorflow-gpu
RUN pip uninstall tensorflow -y
RUN pip install tensorflow tensorflow-datasets tensorflow-hub

# Waiting pytorch to fix the issue with PILLOW_VERSION import
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch

RUN conda install opencv \
    && conda install scikit-image \
    && conda install ipython \
    && conda install matplotlib

# RUN conda install Pillow=6.2.1 

EXPOSE 3000

CMD /bin/bash