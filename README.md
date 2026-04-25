# GPU-Accelerated DPF-Based Private Information Retrieval

## Project Overview

This repository contains a CUDA implementation of DPF-based Private Information Retrieval (PIR), together with small drivers for functional checks and benchmarking. The current artifact provides:

- a CUDA implementation of the main DPF-PIR execution path
- a utility test binary for source-level and helper regression checks
- a functional PIR driver
- a benchmark driver
- a Docker-based build path and a manual build path

## Repository Organization

The source tree is organized as follows.

```text
.
├── CMakeLists.txt
├── Dockerfile
├── emp-ot/
├── emp-tool/
├── mpc_cuda/
│   ├── aes_cuda.h
│   ├── aes_prg_device.h
│   ├── fss_cuda_api.cu
│   ├── fss_cuda_kernels.cu
│   ├── fss_cuda_launch.h
│   ├── mpc_core.h
│   └── pir_context.h
├── mpc_keys/
│   ├── aes_prg_host.h
│   ├── fss_keygen.h
│   └── uint128_type.h
├── test/
│   ├── CMakeLists.txt
│   ├── bench_pir.cpp
│   ├── pir_test_utils.h
│   ├── test_pir.cpp
│   └── test_pir_utils.cpp
└── README.md
```

The main directories are:

- `mpc_cuda/`: CUDA host code, CUDA kernels, launch declarations, and the public PIR interface
- `mpc_keys/`: key-generation support code and shared utility types
- `test/`: artifact entry points for functional testing, benchmarking, and small regression checks
- `emp-tool/`, `emp-ot/`: bundled dependency sources used by the build

## Tested Environment and Dependencies

### Tested environment

The current repository has been exercised in the following environment:

- Ubuntu 22.04
- CUDA 12.4.0
- NVIDIA CUDA Docker base image: `nvidia/cuda:12.4.0-devel-ubuntu22.04`
- CMake 3.22.1 or newer
- GCC/G++ 11 or newer

### Required dependencies

System packages used by the current build flow:

- `build-essential`
- `ca-certificates`
- `cmake`
- `curl`
- `git`
- `libeigen3-dev`
- `libgmp-dev`
- `libmpfr-dev`
- `libssl-dev`
- `pkg-config`
- `python3-pip`
- `wget`

CUDA and GPU requirements:

- CUDA toolkit compatible with CUDA 12.x
- an NVIDIA GPU with a supported driver
- a GPU architecture compatible with the current `CMAKE_CUDA_ARCHITECTURES` setting

Bundled library dependencies:

- `emp-tool` from the bundled `emp-tool/` directory
- `emp-ot` from the bundled `emp-ot/` directory

The build currently sets:

- `CMAKE_CUDA_ARCHITECTURES=89`
- `entry_size=16`
- `NUM_STREAMS=10`
- `NUM_CHUNKS=32768`

If you target a GPU generation other than architecture 89, update `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` before rebuilding.

## Build Instructions

### Option 1: Build with Docker

Requirements:

- Docker
- NVIDIA driver
- NVIDIA Container Toolkit

Build the Docker image from the repository root:

```bash
docker build -t gpu-pir-artifact .
```

Run the image:

```bash
docker run --rm -it --gpus all gpu-pir-artifact
```

If you want to mount the local checkout:

```bash
docker run --rm -it --gpus all -v "$(pwd)":/workspace gpu-pir-artifact
```

Inside the container, the project is already built under `/workspace/build`.

### Option 2: Manual build

Install system packages:

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

Build and install the bundled EMP dependencies:

```bash
cmake -S emp-tool -B emp-tool/build
cmake --build emp-tool/build -j"$(nproc)"
sudo cmake --install emp-tool/build

cmake -S emp-ot -B emp-ot/build
cmake --build emp-ot/build -j"$(nproc)"
sudo cmake --install emp-ot/build
```

If the EMP packages are not discovered automatically, export:

```bash
export CMAKE_PREFIX_PATH="/usr/local/lib/cmake/emp-tool:/usr/local/lib/cmake/emp-ot:${CMAKE_PREFIX_PATH}"
```

Then build this repository:

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

## How to Run the Artifact

The build produces three binaries under `build/bin`.

### 1. Utility regression checks

```bash
./build/bin/test_pir_utils
```

This binary checks helper logic and some source-level repository invariants. It does not require a visible CUDA device.

### 2. Functional PIR driver

```bash
./build/bin/test_pir [n] [batch_size]
```

Examples:

```bash
./build/bin/test_pir
./build/bin/test_pir 24 512
./build/bin/test_pir 20 128
```

Default values:

- `n = 24`
- `batch_size = 512`

### 3. Benchmark driver

```bash
./build/bin/bench_pir [n] [batch_size]
```

Examples:

```bash
./build/bin/bench_pir
./build/bin/bench_pir 22 512
./build/bin/bench_pir 20 128
```

Default values:

- `n = 22`
- `batch_size = 512`

## Configuration Options

The artifact can be configured at two levels.

### Runtime configuration

The functional and benchmark drivers accept:

- `n`: problem size parameter
- `batch_size`: number of PIR queries processed together

Current runtime constraints enforced by the drivers:

- `n` must be at least 8
- for the LUT path, if `n >= 25`, then `batch_size` must be 1

### Build-time configuration

The current build uses the following fixed settings:

- `CMAKE_CUDA_ARCHITECTURES=89` in `CMakeLists.txt`
- `entry_size=16` in `test/CMakeLists.txt`
- `NUM_STREAMS=10` in `test/CMakeLists.txt`
- `NUM_CHUNKS=32768` in `test/CMakeLists.txt`

You can change these values and rebuild if you want to evaluate other GPU targets or compile-time constants.

## How to Interpret the Output

### `test_pir_utils`

Expected successful behavior:

- no output
- exit code `0`

If a check fails, the binary prints a short diagnostic message and exits with a non-zero status.

### `test_pir`

Expected output on success with a visible CUDA device:

- `[PASS] test_pir`
- `[PASS] test_pir_pipeline`
- `[PASS] test_pir_LUT`

If a correctness issue is detected, the program prints a line starting with `[FAIL]` and includes a mismatch description.

If no CUDA device is available, the program prints:

- `[SKIP] test_pir requires a CUDA-capable device`

and exits cleanly.

### `bench_pir`

Expected output on success with a visible CUDA device:

- one timing line per benchmarked mode
- one throughput line per benchmarked mode

Typical labels are:

- `DPF-PIR`
- `DPF-PIR pipeline`
- `DPF-PIR LUT`

If the initial smoke check fails, the program prints a line starting with `[FAIL]`.

If no CUDA device is available, the program prints:

- `[SKIP] bench_pir requires a CUDA-capable device`

and exits cleanly.
