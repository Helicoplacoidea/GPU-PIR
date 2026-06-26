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

Build-essential packages (typically pre-installed on development machines, but listed for completeness):

- `build-essential`, `cmake`, `git`, `pkg-config`

Libraries that may need manual installation:

- `libssl-dev` — required by emp-tool (OpenSSL)
- `libeigen3-dev`, `libgmp-dev`, `libmpfr-dev` — required by emp-ot

CUDA and GPU requirements:

- CUDA toolkit 12.x installed and **visible in `PATH`** (see [Configure CUDA environment](#configure-cuda-environment) below)
- an NVIDIA GPU with a supported driver
- a GPU architecture compatible with the CUDA code; the build auto-detects this by default (see [GPU architecture](#gpu-architecture) below)

Bundled library dependencies:

- `emp-tool` from the bundled `emp-tool/` directory
- `emp-ot` from the bundled `emp-ot/` directory

### Configure CUDA environment

The CUDA toolkit must be discoverable at configure time. Verify that `nvcc` is on `PATH`:

```bash
nvcc --version
```

If `nvcc` is not found, add the CUDA toolkit to your environment. The typical install location is `/usr/local/cuda`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Add these lines to your `~/.bashrc` (or equivalent shell profile) to make the setting persistent.

If you prefer not to modify `PATH`, you can pass the compiler directly to CMake:

```bash
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### GPU architecture

By default, the build automatically detects the GPU architecture via `nvidia-smi` at configure time and sets `CMAKE_CUDA_ARCHITECTURES` accordingly. You will see a log line like:

```
-- Auto-detected GPU architecture: sm_90
```

If `nvidia-smi` is not available or returns no GPU info, the build falls back to CMake's built-in `native` detection.

To override the auto-detected value (e.g., when cross-compiling for a different GPU), pass it explicitly:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90
```

Common architecture numbers:

| Value | GPU generation | Examples |
|-------|---------------|----------|
| 80    | Ampere        | A100, A10 |
| 86    | Ampere        | RTX 3090, A40 |
| 89    | Ada Lovelace  | RTX 4090, L4, L40 |
| 90    | Hopper        | H100, H20 |

Using a mismatched architecture value (e.g., compiling for sm_89 but running on a Hopper GPU) will cause a runtime error: `the provided PTX was compiled with an unsupported toolchain`.

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

> **Note:** When using Docker, the GPU is not visible during `docker build`, so the `nvidia-smi` auto-detection will fail. The Dockerfile uses `ARG CUDA_ARCH=89` (Ada Lovelace) as a default. Override it to match your target GPU:
>
> ```bash
> docker build --build-arg CUDA_ARCH=90 -t gpu-pir-artifact .
> ```
>
> Common values: `80` (A100), `86` (RTX 3090, A40), `89` (RTX 4090, L4, L40), `90` (H100, H20).

### Option 2: Manual build

1. Install required packages (skip any that are already present on your system):

    ```bash
    sudo apt update
    sudo apt install -y build-essential cmake git pkg-config libssl-dev libeigen3-dev libgmp-dev libmpfr-dev
    ```

2. Ensure the CUDA toolkit is on `PATH` (see [Configure CUDA environment](#configure-cuda-environment)).

3. Build and install the bundled EMP dependencies:

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

4. Build this repository:

    ```bash
    cmake -S . -B build
    cmake --build build -j"$(nproc)"
    ```

    If `nvcc` is not on `PATH`, use:

    ```bash
    cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    ```

    If you need to target a specific GPU architecture:

    ```bash
    cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90
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

- **GPU architecture** — controlled by `CMAKE_CUDA_ARCHITECTURES` (see [GPU architecture](#gpu-architecture)). Defaults to auto-detection via `nvidia-smi`.
- **Compile-time constants** — defined in `test/CMakeLists.txt`:
  - `entry_size=16`
  - `NUM_STREAMS=10`
  - `NUM_CHUNKS=32768`

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