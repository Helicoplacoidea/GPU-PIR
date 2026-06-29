#!/usr/bin/env bash
#
# run_bench_pir.sh — 测试流水线 (pipeline) 与非流水线 (non-pipeline) PIR 实现的
# 吞吐量与显存占用, 解析后生成 CSV 与两张 vs-n 图。
#
#   n      = 19, 20, 21, 22, 23, 24
#   batch  = 512
#
#   输出:
#     bench_results_pir/results.csv   (n, mode, throughput, memory_mb)
#     bench_results_pir/throughput_vs_n.png   (pipeline vs non-pipeline)
#     bench_results_pir/memory_vs_n.png       (pipeline vs non-pipeline)
#     bench_results_pir/run_log.txt
#
# 备注: bench_pir 二进制会三连跑 (普通/pipeline/LUT); 本脚本只解析前两种 (普通=
#       non-pipeline, pipeline=DPF-PIR pipeline), LUT 段被忽略。普通 PIR 路径显存计入
#       total_cuda_malloc 但排除 db; pipeline 路径该统计口径几乎不含 db 大头, 故其
#       memory 数值极小, 论文引用时需注明统计口径。
#
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${REPO_DIR}/build/test/bench_pir"
OUT_DIR="${REPO_DIR}/bench_results_pir"
RAW_LOG="${OUT_DIR}/run_log.txt"
CSV="${OUT_DIR}/results.csv"

N_LIST="${1:-19 20 21 22 23 24}"
BATCH="${2:-512}"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${BENCH}" ]]; then
    echo "[ERROR] bench binary not found: ${BENCH}" >&2
    echo "        请先 cmake --build build" >&2
    exit 1
fi

echo "[INFO] bench: ${BENCH}"
echo "[INFO] n: ${N_LIST}"
echo "[INFO] batch_size: ${BATCH}"
echo "[INFO] output dir: ${OUT_DIR}"
echo

python3 - "${BENCH}" "${OUT_DIR}" "${RAW_LOG}" "${CSV}" "${BATCH}" ${N_LIST} <<'PYEOF'
import os, re, subprocess, sys, time
from datetime import datetime

bench, out_dir, raw_log, csv_path, batch = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
n_values = [int(x) for x in sys.argv[6:]]

MODES = ["DPF-PIR", "DPF-PIR pipeline"]  # 只关心 non-pipeline 与 pipeline

def parse_run(text):
    lines = text.splitlines()
    seg_starts = [i for i,l in enumerate(lines) if l.startswith("CUDA memory initialized")]
    results = {}
    for idx,s in enumerate(seg_starts):
        seg_end = seg_starts[idx+1] if idx+1<len(seg_starts) else len(lines)
        mem=None
        for l in lines[s:seg_end]:
            m=re.search(r"Total allocated CUDA memory \(excluding database\):\s*([\d.]+)\s*MB",l)
            if m: mem=float(m.group(1))
        mode_label=t_ms=thr=None
        for l in lines[s:seg_end]:
            mt=re.match(r"^(.+?)\s+Time taken:\s*([\d.eE+-]+)\s*ms",l)
            if mt: mode_label=mt.group(1).strip(); t_ms=float(mt.group(2))
            mh=re.search(r"Throughput:\s*([\d.eE+-]+)\s*pirs/s",l)
            if mh: thr=float(mh.group(1))
        if mode_label in MODES:
            results[mode_label]={"time_ms":t_ms,"throughput":thr,"memory_mb":mem}
    return results

rows=[]
with open(raw_log,"w") as logf:
    logf.write(f"# PIR bench {datetime.now().isoformat(timespec='seconds')}  batch_size={batch}  n={n_values}\n")
    for n in n_values:
        t0=time.time()
        proc=subprocess.run([bench,str(n),str(batch)],capture_output=True,text=True,timeout=3600)
        wall=time.time()-t0
        logf.write(f"\n===== n={n} batch={batch} (exit={proc.returncode}, wall={wall:.1f}s) =====\n")
        logf.write(proc.stdout)
        if proc.stderr: logf.write("--- stderr ---\n"+proc.stderr)
        logf.flush()
        if proc.returncode!=0:
            print(f"[WARN] n={n} exit={proc.returncode}, skipped (see {raw_log})")
            continue
        res=parse_run(proc.stdout)
        for mode in MODES:
            r=res.get(mode)
            if r and r["time_ms"] is not None:
                rows.append({"n":n,"mode":mode,"throughput":r["throughput"],"memory_mb":r["memory_mb"]})
                print(f"  n={n:<3} {mode:<18} thrpt={r['throughput']:.1f} pirs/s  mem={r['memory_mb']} MB")
            else:
                print(f"  n={n:<3} {mode:<18} [未解析到结果, 见 {raw_log}]")

import csv as csvmod
with open(csv_path,"w",newline="") as f:
    w=csvmod.DictWriter(f,fieldnames=["n","mode","throughput","memory_mb"])
    w.writeheader()
    for r in rows: w.writerow(r)
print(f"\n[INFO] CSV -> {csv_path}  ({len(rows)} rows)")

import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if not rows:
    print("[WARN] 无数据, 跳过画图"); sys.exit(0)

df=pd.DataFrame(rows)
mode_map={"DPF-PIR":"Non-pipeline","DPF-PIR pipeline":"Pipeline"}
df["mode"]=df["mode"].map(mode_map)

def _save(fig,name):
    p=os.path.join(out_dir,name)
    fig.tight_layout(); fig.savefig(p,dpi=150); plt.close(fig)
    print(f"[INFO] figure -> {p}")

# 1) 吞吐量 vs n
fig,ax=plt.subplots(figsize=(7,4.5))
for mod,mk,col in [("Non-pipeline","o","tab:blue"),("Pipeline","s","tab:red")]:
    d=df[df["mode"]==mod].sort_values("n")
    ax.plot(d["n"],d["throughput"],marker=mk,color=col,label=mod,linewidth=2)
for _,r in df.iterrows():
    ax.annotate(f"{r['throughput']:.0f}",(r["n"],r["throughput"]),textcoords="offset points",xytext=(0,7),fontsize=7,ha="center")
ax.set_xlabel("n (log2 database size)"); ax.set_ylabel("Throughput (PIRs/s)")
ax.set_title(f"PIR Throughput vs n  (batch_size={batch})")
ax.set_xticks(sorted(df["n"].unique())); ax.grid(alpha=.3); ax.legend()
_save(fig,"throughput_vs_n.png")

# 2) 显存 vs n
fig,ax=plt.subplots(figsize=(7,4.5))
for mod,mk,col in [("Non-pipeline","o","tab:blue"),("Pipeline","s","tab:red")]:
    d=df[df["mode"]==mod].sort_values("n")
    ax.plot(d["n"],d["memory_mb"],marker=mk,color=col,label=mod,linewidth=2)
ax.set_xlabel("n (log2 database size)"); ax.set_ylabel("CUDA memory (MB, excludes database)")
ax.set_title(f"PIR CUDA Memory vs n  (batch_size={batch})")
ax.set_xticks(sorted(df["n"].unique())); ax.grid(alpha=.3); ax.legend()
_save(fig,"memory_vs_n.png")

print("\n[DONE] 全部完成。")
PYEOF

echo
echo "=== 产物清单 ==="
ls -la "${OUT_DIR}"
