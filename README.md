# GPU-Accelerated DPF-Based Private Information Retrieval (Serving Variants)

## Project Overview

This repository builds on the artifact released with the CHES paper at
[`Crypto-Lab-WHU/GPU-PIR`](https://github.com/Crypto-Lab-WHU/GPU-PIR) and adds a
serving-oriented redesign on top of that DPF-PIR implementation.

The additions are:

- an optimized GGM tree evaluation that folds selection-vector generation into
  the final tree layer, so the serving paths no longer materialise intermediate
  seed/bit buffers or run a separate packing step;
- a **low-latency** kernel path for a single query, which spreads the database
  scan over all SMs using sector-view conditional loads;
- a **high-throughput** kernel path for batched queries, which tiles the
  database into shared memory and reuses each tile across the whole batch
  (synchronous or `cp.async` double-buffered loading).

Both paths keep the same row-major database layout, the same DPF key format, and
the same public API shape as the original artifact.

## Design

```text
                     Optimized GGM tree evaluation
                                  │
                       generate selection shares
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
        single query / small batch        medium / large batch
                  │                               │
      multi-SM sector-view loads        shared-memory DB tiling
        (only selected rows)                (128-entry tiles)
                  │                               │
     per-block partials + reduce       sync / async load & reuse
                  │                               │
           low execution latency            high throughput
```

### Optimized GGM tree evaluation

The original flow evaluated the whole GGM tree, wrote the final `(s, t)` nodes,
and then packed them into the selection-vector shares `se`. This repository fuses
the two steps: the final tree layer applies the leaf block directly and XORs it
into `se` while expanding (`EvalAll_BlockEvaluation_Se`). As a consequence:

- the single-query and batch paths no longer allocate the `s_res` / `t_res`
  intermediate buffers and no longer run a separate selection-generation pass;
  the pipeline and LUT paths keep the original two-step flow;
- the fused kernel is the shared prerequisite for both serving paths;
- `test_fused_eval` checks the fused output against a CPU reference over edge-case
  indices, single-query repeats, and both prefix/final-layer splits (`n = 8..20`,
  batch up to 129).

### Low-latency path: multi-SM sector-view conditional access

For a single query (`N == 1`), the database is partitioned into contiguous row
ranges, one CUDA block per range, and the number of partitions is derived from
the device's SM count (capped by the number of 128-row groups). Each block only
loads rows whose selection bit is set, emits one partial per block, and a small
second kernel XOR-reduces the partials. The selected kernel
(`sector_mul_red4_kernel`) assigns four lanes per row, so each lane reads 64
contiguous bytes (two 32-byte sectors) and eight rows advance per warp. With the
row-major layout a selected row costs exactly its payload (1.0x); the column-major
variant is kept for comparison but transfers about 1.5x at 50% row density.

### High-throughput path: shared-memory database tiling

For batched queries (`init_pir_batch`), each block owns a contiguous database
range and walks it in tiles of 128 entries. A tile is staged to shared memory
once and consumed by all queries in the batch (one thread per query), and a single
128-bit `se` word covers a whole tile, so no per-entry selection loads are
needed. Two variants are provided:

- **sync** (`BatchMulRedSync`): one shared-memory tile, blocking loads;
- **async** (`BatchMulRedAsync`): double-buffered tiles prefetched with
  `cp.async` (`cuda_pipeline`).

Both variants write `[block][query][component]` partials and share one reduce
kernel. The batch path uses its own context (`PirBatchContext`), so its CUDA
memory report does not include the row-major `blocksum`.

### Choosing a path

- one query at a time, latency matters → low-latency path (`init_pir` with
  `N = 1`);
- many concurrent queries, throughput matters → batch path (`init_pir_batch`
  with `batch_size` queries, sync or async);
- `bench_serving` measures both on the same database and prints the latency
  percentiles, the batched throughput, and the per-request throughput ratio.

## Repository Organization

```text
.
├── CMakeLists.txt
├── Dockerfile
├── emp-ot/                      bundled dependency
├── emp-tool/                    bundled dependency
├── mpc_cuda/
│   ├── aes_cuda.h
│   ├── aes_prg_device.h
│   ├── batch_tiled.cu           high-throughput tiled kernels
│   ├── dcf_keys.h
│   ├── fss_cuda_api.cu          host-side PIR contexts and entry points
│   ├── fss_cuda_kernels.cu      DPF keygen and fused GGM tree evaluation
│   ├── fss_cuda_launch.h        kernel declarations
│   ├── mpc_core.h
│   ├── pir_config.h             entry_size, tile size
│   ├── pir_context.h            PirContext, PirBatchContext, LUT/pipeline contexts
│   └── sector_view.cu           low-latency sector-view kernels
├── mpc_keys/                    key-generation support and shared types
├── test/
│   ├── CMakeLists.txt
│   ├── bench_lut_only.cpp
│   ├── bench_pir.cpp
│   ├── bench_sector_view.cu     single-query kernel study
│   ├── bench_serving.cpp        low-latency vs high-throughput study
│   ├── pir_test_utils.h
│   ├── test_fused_eval.cu       fused selection generation vs CPU reference
│   ├── test_pir.cpp             functional PIR driver
│   └── test_pir_utils.cpp       helper/source-level regression checks
├── run_bench_pir.sh
├── run_bench_lut.sh
└── README.md
```

## Tested Environment and Dependencies

### Tested environment

- Ubuntu 22.04
- CUDA 12.4.0 (`nvidia/cuda:12.4.0-devel-ubuntu22.04`)
- CMake 3.25 or newer
- GCC/G++ 11 or newer

### Required dependencies

- `build-essential`, `cmake`, `git`, `pkg-config`
- `libssl-dev` (required by emp-tool)
- `libeigen3-dev`, `libgmp-dev`, `libmpfr-dev` (required by emp-ot)
- CUDA toolkit 12.x visible in `PATH`
- an NVIDIA GPU with a supported driver

### Configure CUDA environment

```bash
nvcc --version
```

If `nvcc` is not on `PATH`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

or pass it to CMake directly:

```bash
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### GPU architecture

The build auto-detects the architecture via `nvidia-smi` at configure time
(`-- Auto-detected GPU architecture: sm_90`) and falls back to CMake's `native`
detection when no GPU is visible. Override it when needed:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
```

| Value | Generation | Examples |
|-------|------------|----------|
| 80    | Ampere     | A100, A10 |
| 86    | Ampere     | RTX 3090, A40 |
| 89    | Ada        | RTX 4090, L4, L40 |
| 90    | Hopper     | H100, H20 |

A mismatched value fails at runtime with `the provided PTX was compiled with an
unsupported toolchain`.

## Build Instructions

### Option 1: Docker

```bash
docker build -t gpu-pir-artifact .
docker run --rm -it --gpus all -v "$(pwd)":/workspace gpu-pir-artifact
```

The GPU is not visible during `docker build`, so the image defaults to
`ARG CUDA_ARCH=89`. Override it for your target:

```bash
docker build --build-arg CUDA_ARCH=90 -t gpu-pir-artifact .
```

### Option 2: Manual build

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config \
    libssl-dev libeigen3-dev libgmp-dev libmpfr-dev

cmake -S emp-tool -B emp-tool/build && cmake --build emp-tool/build -j"$(nproc)"
sudo cmake --install emp-tool/build

cmake -S emp-ot -B emp-ot/build && cmake --build emp-ot/build -j"$(nproc)"
sudo cmake --install emp-ot/build

cmake -S . -B build
cmake --build build -j"$(nproc)"
```

If the EMP packages are not found, export:

```bash
export CMAKE_PREFIX_PATH="/usr/local/lib/cmake/emp-tool:/usr/local/lib/cmake/emp-ot:${CMAKE_PREFIX_PATH}"
```

## How to Run

The build produces the binaries below under `build/test`. The GPU drivers skip
cleanly with a `SKIP` line when no CUDA device is visible; `test_pir_utils` does
not need a device.

| Binary | Purpose | Usage | Defaults |
|--------|---------|-------|----------|
| `test_pir_utils` | helper and source-level regression checks | `./build/test/test_pir_utils` | — |
| `test_fused_eval` | fused selection generation vs CPU reference | `./build/test/test_fused_eval` | fixed case matrix |
| `test_pir` | functional PIR, batch sync/async, pipeline, LUT | `./build/test/test_pir [n] [batch_size]` | `24`, `512` |
| `bench_pir` | row-major, batch sync/async, pipeline and LUT throughput | `./build/test/bench_pir [n] [batch_size]` | `22`, `512` |
| `bench_sector_view` | single-query kernel variants from one SM to multi-SM | `./build/test/bench_sector_view [n] [samples]` | `20`, `31` |
| `bench_serving` | low-latency vs high-throughput comparison | `./build/test/bench_serving [n] [batch] [low_samples] [batch_samples]` | `20`, `512`, `31`, `5` |
| `bench_lut_only` | LUT-only throughput | `./build/test/bench_lut_only [n] [batch_size]` | `20`, `512` |

Examples:

```bash
./build/test/test_pir 24 512
./build/test/bench_sector_view 24 11
./build/test/bench_serving 24 512 51 7
```

### Public API

```cpp
// Low-latency path: N == 1 uses the multi-SM sector-view kernels.
PirContext ctx = init_pir(N, n, db);
uint128_t *out = test_dpf_pir(key, ctx, N);
free_cuda_memory(ctx);

// High-throughput path: batch of queries, mode 0 = sync, 1 = async.
pir_batch_context bctx = init_pir_batch(batch_size, n, db);
uint128_t *out = test_dpf_pir_batch(key, bctx, batch_size, mode);
free_cuda_memory(bctx);
```

`init_pir_batch` allocates the batch partials/output buffers and configures the
async kernel's opt-in shared memory; the row-major and batch paths are
benchmarked with separate contexts because their memory footprints differ.

## How to Interpret the Output

### `test_pir_utils`

Expected: no output, exit code `0`. A failed check prints a diagnostic and exits
non-zero.

### `test_fused_eval`

Expected: one `PASS n=... B=...` line per case. Exit code `77` (skip) when no
CUDA device is available.

### `test_pir`

Expected on success with a visible device:

```text
[PASS] test_pir
[PASS] test_pir_single
[PASS] test_pir_batch_sync
[PASS] test_pir_batch_async
[PASS] test_pir_pipeline
[PASS] test_pir_LUT
```

Failures print a `[FAIL]` line with a mismatch description; a missing device
prints `[SKIP] test_pir requires a CUDA-capable device`.

### `bench_sector_view`

Prints the device line, the per-party selection density, the isolated selection
generation time, and one `RESULT` line per variant:

```text
RESULT n=... variant=multi_sector4_p... median_ms=... min_ms=... p10_ms=... p90_ms=...
```

Variants cover the baseline single-SM kernel (`native_1sm`), multi-SM row,
sector, sector4 and column-major kernels, and the batched shared-memory
sync/async kernels at `batch = 1`. Raw per-sample timings are emitted as
`SAMPLE,n,variant,ms` lines for plotting.

### `bench_serving`

Prints the device line, then:

```text
LOW_LATENCY n=... samples=... p50_ms=... p95_ms=... p99_ms=... min_ms=... max_ms=...
THROUGHPUT n=... B=... mode=sync  samples=... median_ms=... p99_ms=... pirs_per_s=... per_request_us=...
THROUGHPUT n=... B=... mode=async samples=... median_ms=... p99_ms=... pirs_per_s=... per_request_us=...
COMPARISON n=... single_p50_ms=... single_p99_ms=... batch_async_p50_ms=... batch_async_p99_ms=... async_per_request_us=... throughput_ratio=...
```

Both paths are verified against the reconstructed database before timing;
`throughput_ratio` compares batched async throughput with single-query
throughput at the same `n`.

### `bench_pir`

Prints one timing and one throughput line for each mode:

```text
DPF-PIR              (row-major)
DPF-PIR batch sync
DPF-PIR batch async
DPF-PIR pipeline
DPF-PIR LUT
```

## Constraints

- `n >= 8`; the packed DPF depth is `maxlayer = n - 7`.
- Batch path: `1 <= batch_size <= 512`. The tiled kernels use one thread per
  query and hold `entry_size` `uint128_t` accumulators per thread.
- Shared memory per batch block: 32 KB (`sync`) or 64 KB (`async`,
  double-buffered). `init_pir_batch` checks the device limit and requests the
  opt-in size.
- `bench_serving` accepts `n = 8..24`, `batch <= 512` and `batch <= 2^n`.
- Compile-time constants: `entry_size = 16` and `k_batch_tile_entries = 128` in
  `mpc_cuda/pir_config.h`; `NUM_STREAMS = 10` and `NUM_CHUNKS = 32768` in
  `test/CMakeLists.txt`.

## Relation to the Original Artifact

This repository reuses the DPF construction, key layout, database format and the
pipeline/LUT paths from [`Crypto-Lab-WHU/GPU-PIR`](https://github.com/Crypto-Lab-WHU/GPU-PIR),
the artifact of the accompanying CHES paper. The optimized GGM tree evaluation
and the two serving kernel families described above are the additions here; the
original paper figures and tables are not reproduced by this repository.

## License

See [LICENSE](LICENSE).
