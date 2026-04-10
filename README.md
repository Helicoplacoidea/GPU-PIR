# 🚀 fss-cuda: Fast PIR with CUDA Acceleration

This project implements a Private Information Retrieval (PIR) protocol using C++ and CUDA, with all dependencies encapsulated in a Docker environment. It is designed for easy deployment and GPU acceleration out of the box.

## 📁 Project Structure

 ├── CMakeLists.txt        # Top-level CMake configuration
 ├── Dockerfile_Gen        # Dockerfile to build the CUDA development image
 ├── build/                # Build directory (generated automatically)
 ├── test/                 # Test cases
 ├── emp-ot/                 # 
 ├── emp-tool/                # 
 ├── mpc_cuda/             # CUDA kernel source code
 ├── mpc_keys/             # Cryptographic utilities
 └── README.md             # This file

---

## 🚀 Getting Started with Docker(Recommended)

### ✅ 1. Build the Docker Image (only once)

Make sure the following are installed:

- Docker (v20.10+ recommended)
- NVIDIA GPU Driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Build the image:

```bash
docker build -f Dockerfile_Gen -t fss-cuda-image .
```

It will be very time-consuming to obtain the CUDA image for the first time

### ✅ 2. Run the program on docker

Enter the image:

```bash
docker run --rm -it --gpus all fss-cuda-image
```

Run the test/bench program

```bash
./build/bin/test_pir
```

```bash
./build/bin/bench_pir
```

You can also mount your code directory if you want to persist changes:

```bash
docker run --rm -it --gpus all -v $(pwd):/workspace fss-cuda-image
```

## 🛠️ Manual Setup Without Docker

If you prefer not to use Docker, follow the steps below to set up your environment manually:

### 1. System Requirements

- **Ubuntu 20.04 / 22.04**
- **CUDA Toolkit 11.8+**
- **NVIDIA GPU + Driver**
- **CMake ≥ 3.22**
- **GCC/G++ ≥ 9**
- **Git**

### 2. Install Dependencies

```bash
sudo apt update && apt install -y \
    build-essential git cmake wget curl \
    libssl-dev libgmp-dev libmpfr-dev libeigen3-dev \
    python3-pip pkg-config
```

### 3. Clone and Build Required **Libraries**

Build emp-ot/emp-tool:

```bash
mkdir /emp-tool/build && cd /emp-tool/build && \
    cmake .. && make -j$(nproc) && make install
mkdir /emp-ot/build && cd /emp-ot/build && \
    cmake .. && make -j$(nproc) && make install
```

Build fss-cuda:

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

Run the test/bench program:

```bash
./bin/test_pir
```

```bash
./bin/bench_pir
```

## 📌 Notes

- Make sure the `CUDA_ARCH` matches your GPU (e.g., `sm_89` for RTX 4090).
- You can configure parameters like `entry_size`, `NUM_CHUNKS`, etc., in CMake or `target_compile_definitions`.

