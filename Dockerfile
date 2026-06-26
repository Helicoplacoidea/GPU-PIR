FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# GPU architecture to compile for. Override with: docker build --build-arg CUDA_ARCH=90 ...
ARG CUDA_ARCH=89

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libeigen3-dev \
    libgmp-dev \
    libmpfr-dev \
    libssl-dev \
    pkg-config \
    python3-pip \
    wget && \
    rm -rf /var/lib/apt/lists/*

COPY emp-tool /workspace/emp-tool
RUN cmake -S /workspace/emp-tool -B /workspace/emp-tool/build && \
    cmake --build /workspace/emp-tool/build -j"$(nproc)" && \
    cmake --install /workspace/emp-tool/build

COPY emp-ot /workspace/emp-ot
RUN cmake -S /workspace/emp-ot -B /workspace/emp-ot/build && \
    cmake --build /workspace/emp-ot/build -j"$(nproc)" && \
    cmake --install /workspace/emp-ot/build

ENV CMAKE_PREFIX_PATH="/usr/local/lib/cmake/emp-tool:/usr/local/lib/cmake/emp-ot:${CMAKE_PREFIX_PATH}"

COPY . /workspace

RUN rm -rf /workspace/build && \
    cmake -S /workspace -B /workspace/build -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} && \
    cmake --build /workspace/build -j"$(nproc)"

CMD ["/bin/bash"]
