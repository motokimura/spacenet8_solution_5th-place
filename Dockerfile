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

ENV PYTHONPATH $PYTHONPATH:/work

WORKDIR /work
