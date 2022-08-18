FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND="noninteractive" TZ="Asia/Tokyo" apt-get install -y \
    postgresql-client \
    libpq-dev \
    gdal-bin \
    libgdal-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

# install segmentation_models_pytorch
RUN git clone https://github.com/qubvel/segmentation_models.pytorch.git /smp
WORKDIR /smp
RUN git checkout 740dab561ccf54a9ae4bb5bda3b8b18df3790025 && pip install .

# copy files
COPY configs /work/configs
COPY scripts /work/scripts
COPY spacenet8_model /work/spacenet8_model
COPY tools /work/tools
COPY *.sh /work/

RUN chmod a+x /work/scripts/*.sh
RUN chmod a+x /work/scripts/p3/*.sh
RUN chmod a+x /work/*.sh

# download motokimura's home-built models
# these models are removed before training (see train.sh)
WORKDIR /work/models
# effnet-b5 foundation
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50000.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50001.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50002.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50003.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50004.zip
# effnet-b5 flood
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50010.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50011.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50012.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50013.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50014.zip
# effnet-b6 foundation
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60400.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60401.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60402.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60403.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60404.zip
# effnet-b6 flood
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60420.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60421.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60422.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60423.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60424.zip
RUN unzip "*.zip"

ENV PYTHONPATH $PYTHONPATH:/work

WORKDIR /work
