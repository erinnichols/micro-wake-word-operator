FROM --platform=linux/amd64 python:3.12-slim

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        espeak-ng git build-essential cmake ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel cython

RUN pip install numpy==1.26.4

RUN pip install piper-tts piper-sample-generator

# Clone piper and build monotonic_align Cython extension
RUN git clone --depth 1 https://github.com/rhasspy/piper /tmp/piper && \
    pip install -e /tmp/piper/src/python --no-deps && \
    cd /tmp/piper/src/python && \
    python3 piper_train/vits/monotonic_align/setup.py build_ext --inplace && \
    mkdir -p piper_train/vits/monotonic_align/monotonic_align && \
    cp piper_train/vits/monotonic_align/core*.so piper_train/vits/monotonic_align/monotonic_align/ && \
    touch piper_train/vits/monotonic_align/monotonic_align/__init__.py

RUN pip install tensorflow tensorboard

RUN git clone https://github.com/OHF-Voice/micro-wake-word.git /tmp/microwakeword-src && \
    pip install -e /tmp/microwakeword-src

RUN pip install librosa soundfile fsspec datasets scipy mmap_ninja torchcodec

RUN pip install audiomentations --upgrade

EXPOSE 6006

WORKDIR /workspace
