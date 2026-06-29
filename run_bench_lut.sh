#!/usr/bin/env bash
#
# run_bench_lut.sh — 只测 DPF-PIR LUT 的吞吐 vs batch_size。
#
#   n      = 16, 18, 20, 24  (均 <25, 满足 LUT 的 n 约束)
#   batch  = 1, 10, 100, 1000, 10000   (对数刻度横轴)
#
#   输出:
#     bench_results_lut/results.csv
#     bench_results_lut/lut_n16_n18.png   (n=16 与 n=18 一张图)
#     bench_results_lut/lut_n20_n24.png   (n=20 与 n=24 一张图)
#     bench_results_lut/run_log.txt
#
# 备注: batch 曾计划测到 1e5, 但 n=24 在 1e5 时 LUT 路径需 ~419GB 显存
#       (d_s_res ≈ batch * 2^(n-2) 字节级累加), 物理不可行, 故上限降到 1e4。
#
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${REPO_DIR}/build/test/bench_lut_only"
OUT_DIR="${REPO_DIR}/bench_results_lut"
RAW_LOG="${OUT_DIR}/run_log.txt"
CSV="${OUT_DIR}/results.csv"

N_LIST="${1:-16 18 20 24}"
BATCH_LIST="${2:-1 10 100 1000 10000}"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${BENCH}" ]]; then
    echo "[ERROR] LUT bench binary not found: ${BENCH}" >&2
    echo "        请先 cmake --build build --target bench_lut_only" >&2
    exit 1
fi

echo "[INFO] lut bench: ${BENCH}"
echo "[INFO] n: ${N_LIST}"
echo "[INFO] batch: ${BATCH_LIST}"
echo "[INFO] output dir: ${OUT_DIR}"
echo

python3 - "${BENCH}" "${OUT_DIR}" "${RAW_LOG}" "${CSV}" ${N_LIST} ${BATCH_LIST} <<'PYEOF'
import os, re, subprocess, sys, time
from datetime import datetime

bench, out_dir, raw_log, csv_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
n_values = [int(x) for x in sys.argv[5:5+4]]  # 取前4个为 n
# 剩余为 batch (n 与 batch 个数固定, 但为稳妥按规则区分: 这里 argv 形如
#   script bench out raw csv n1 n2 n3 n4 b1 b2 b3 b4 b5
# )
# 实际上我们直接约定: n_values 取索引 [5:9], batch 取索引 [9:]
batch_values = [int(x) for x in sys.argv[9:]]

def parse_lut(text):
    """bench_lut_only 只输出一段: 'DPF-PIR LUT Time taken: X ms' + 'Throughput: Y pirs/s'"""
    t = re.search(r"DPF-PIR LUT Time taken:\s*([\d.eE+-]+)\s*ms", text)
    th = re.search(r"Throughput:\s*([\d.eE+-]+)\s*pirs/s", text)
    return (float(t.group(1)) if t else None,
            float(th.group(1)) if th else None)

rows = []
with open(raw_log, "w") as logf:
    logf.write(f"# LUT bench {datetime.now().isoformat(timespec='seconds')}  n={n_values}  batch={batch_values}\n")
    for n in n_values:
        for b in batch_values:
            t0 = time.time()
            proc = subprocess.run([bench, str(n), str(b)],
                                  capture_output=True, text=True, timeout=1800)
            wall = time.time() - t0
            logf.write(f"\n===== n={n} batch={b} (exit={proc.returncode}, wall={wall:.1f}s) =====\n")
            logf.write(proc.stdout)
            if proc.stderr:
                logf.write("--- stderr ---\n" + proc.stderr)
            logf.flush()
            combined = proc.stdout + proc.stderr
            # OOM 检测
            if proc.returncode != 0 or "out of memory" in combined.lower():
                print(f"  n={n:<3} batch={b:<7} [失败: exit={proc.returncode}, OOM 或错误, 见 {raw_log}]")
                continue
            t_ms, thr = parse_lut(proc.stdout)
            if t_ms is not None:
                rows.append({"n": n, "batch": b, "time_ms": t_ms, "throughput": thr})
                print(f"  n={n:<3} batch={b:<7} time={t_ms:.3f} ms  thrpt={thr:.1f} pirs/s")
            else:
                print(f"  n={n:<3} batch={b:<7} [未解析到结果, 见 {raw_log}]")

import csv as csvmod
with open(csv_path, "w", newline="") as f:
    w = csvmod.DictWriter(f, fieldnames=["n","batch","time_ms","throughput"])
    w.writeheader()
    for r in rows: w.writerow(r)
print(f"\n[INFO] CSV -> {csv_path}  ({len(rows)} rows)")

import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if not rows:
    print("[WARN] 无数据, 跳过画图"); sys.exit(0)

df = pd.DataFrame(rows)
COLORS = {16:"tab:blue", 18:"tab:orange", 20:"tab:green", 24:"tab:red"}
MARKERS = {16:"o", 18:"s", 20:"^", 24:"D"}

def _plot(ns, name, title):
    fig, ax = plt.subplots(figsize=(7,4.8))
    for n in ns:
        d = df[df["n"]==n].sort_values("batch")
        if d.empty: continue
        ax.plot(d["batch"], d["throughput"], marker=MARKERS.get(n,"o"),
                color=COLORS.get(n,"gray"), label=f"n={n}", linewidth=2)
        for _,r in d.iterrows():
            ax.annotate(f"{r['throughput']:.0f}", (r["batch"], r["throughput"]),
                        textcoords="offset points", xytext=(0,7), fontsize=7, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("batch_size (log scale)")
    ax.set_ylabel("Throughput (PIRs/s)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[INFO] figure -> {p}")

# n=16, 18 一张图; n=20, 24 一张图
_plot([16,18], "lut_n16_n18.png", "DPF-PIR LUT Throughput (n=16, 18)")
_plot([20,24], "lut_n20_n24.png", "DPF-PIR LUT Throughput (n=20, 24)")

print("\n[DONE] 全部完成。")
PYEOF

echo
echo "=== 产物清单 ==="
ls -la "${OUT_DIR}"
