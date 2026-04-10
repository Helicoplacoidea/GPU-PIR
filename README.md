# GPU-Accelerated DPF-Based Private Information Retrieval 

This repository contains a CUDA implementation of DPF-based Private Information Retrieval (PIR) together with small test and benchmark drivers. The current codebase has been cleaned up to remove the historical `cudaDPF/` copy, split the main CUDA implementation into host-side and kernel-side translation units, and expose typed PIR contexts instead of raw `std::vector<void *>` handles.

## Repository Layout

```text
.
├── CMakeLists.txt
├── Dockerfile
├── emp-ot/
├── emp-tool/
├── mpc_cuda/
│   ├── fss_cuda_api.cu
│   ├── fss_cuda_kernels.cu
│   ├── fss_cuda_launch.h
│   ├── mpc_core.h
│   └── pir_context.h
├── mpc_keys/
├── test/
│   ├── bench_pir.cpp
│   ├── pir_test_utils.h
│   ├── test_pir.cpp
│   └── test_pir_utils.cpp
└── README.md
```

## Build with Docker

### Build the image

Requirements:

- Docker
- NVIDIA driver
- NVIDIA Container Toolkit

From the repository root:

```bash
docker build -t fss-cuda .
```

### Run the container

```bash
docker run --rm -it --gpus all fss-cuda
```

If you want to mount the local checkout:

```bash
docker run --rm -it --gpus all -v "$(pwd)":/workspace fss-cuda
```

Inside the container, the project is already built under `/workspace/build`.

Useful commands:

```bash
./build/bin/test_pir_utils
./build/bin/test_pir 24 512
./build/bin/bench_pir 22 512
```

## Manual Build

### Requirements

- Ubuntu 22.04 or a similar recent Linux distribution
- CUDA Toolkit 12.x
- CMake 3.22+
- GCC/G++ 9+
- A GPU supported by your local CUDA installation

### Install system packages

```bash
sudo apt update
sudo apt install -y \
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
    wget
```

### Build bundled dependencies

```bash
cmake -S emp-tool -B emp-tool/build
cmake --build emp-tool/build -j"$(nproc)"
sudo cmake --install emp-tool/build

cmake -S emp-ot -B emp-ot/build
cmake --build emp-ot/build -j"$(nproc)"
sudo cmake --install emp-ot/build
```

If the installed EMP packages are not found automatically, export:

```bash
export CMAKE_PREFIX_PATH="/usr/local/lib/cmake/emp-tool:/usr/local/lib/cmake/emp-ot:${CMAKE_PREFIX_PATH}"
```

### Build this project

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

## Running Tests and Benchmarks

Three binaries are built under `build/bin`:

- `test_pir_utils`: source-level and helper regression checks, does not require a visible CUDA device
- `test_pir [n] [batch_size]`: functional PIR driver
- `bench_pir [n] [batch_size]`: benchmark driver

Examples:

```bash
./build/bin/test_pir_utils
./build/bin/test_pir 24 512
./build/bin/bench_pir 22 512
```

If no CUDA-capable device is visible at runtime, `test_pir` and `bench_pir` print `SKIP` and exit cleanly.

## Current Build Configuration

- CUDA architecture is currently fixed to `89` in `CMakeLists.txt`. Change `CMAKE_CUDA_ARCHITECTURES` if you are targeting a different GPU generation.
- Test targets compile with:
  - `entry_size=16`
  - `NUM_STREAMS=10`
  - `NUM_CHUNKS=32768`
- `test_pir` defaults to `n=24`, `batch_size=512`
- `bench_pir` defaults to `n=22`, `batch_size=512`

## Notes

- The host-side CUDA runtime calls in `mpc_cuda/fss_cuda_api.cu` are wrapped in unified error-check macros.
- The main CUDA implementation is now split so host orchestration and kernel/device code are easier to read and maintain.
- The public API in `mpc_cuda/mpc_core.h` now uses named context structs instead of raw pointer vectors.
