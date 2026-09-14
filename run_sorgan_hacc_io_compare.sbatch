#!/bin/bash
#SBATCH --job-name=hacc-io-cmp
#SBATCH --partition=all
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=sorgan-cpu[1-4],sorgan-gpu[1-4]
#SBATCH --exclude=sorgan-cpu5,sorgan-cpu6
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

# HACC-IO backend compare (single shared allocation, cases serialized):
#   posix / mpiio  : upstream GLEAN driver, N SPMD MPI ranks
#   ompfile-wt     : split-role OMPFILE variant (1 app rank + N-1 proxies),
#                    write-through, fsync-per-write (safe default posture)
#   ompfile-wb     : split-role, readthrough staging + write-back capture +
#                    fsync policy close (tuned posture)
# The OMPFILE variant emulates N logical ranks (HACC_RANKS=N) so aggregate
# bytes match the SPMD baselines — same split-role methodology caveat as the
# IOR backend-compare lane (docs/quick-reference.md).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT:-/scratch/rodrigo.freitas/io-stage/src}}"
if [[ -n "${SIF_IMAGE:-}" && -e "${SIF_IMAGE:-}" ]]; then
  SIF_IMAGE="${SIF_IMAGE}"
elif [[ -e "${REPO_ROOT}/ompc-base_latest.sif" ]]; then
  SIF_IMAGE="${REPO_ROOT}/ompc-base_latest.sif"
elif [[ -e "${REPO_ROOT}/sifs/ompc-base_latest.sif" ]]; then
  SIF_IMAGE="${REPO_ROOT}/sifs/ompc-base_latest.sif"
else
  echo "Error: Apptainer image not found under ${REPO_ROOT}" >&2
  exit 1
fi
APPTAINER_BIN="${APPTAINER_BIN:-/usr/bin/apptainer}"
LLVM_ROOT="${LLVM_ROOT:-${LLVM_BUILD_ROOT:-${REPO_ROOT}/llvm-infra/llvm-builds/apptainer-Debug}}"
LLVM_INSTALL_ROOT="${LLVM_INSTALL_ROOT:-${REPO_ROOT}/llvm-infra/llvm-installs/apptainer-Debug}"
HACC_DIR="${HACC_DIR:-${REPO_ROOT}/application/hacc-io-fork}"
OMPFILE_INC="${OMPFILE_INC:-${REPO_ROOT}/llvm-infra/llvm-project/openmp/libompfile/include}"
OMPFILE_LIB="${OMPFILE_LIB:-${LLVM_ROOT}/runtimes/runtimes-bins/openmp/libompfile}"
OMP_RUNTIME_LIB="${OMP_RUNTIME_LIB:-${LLVM_ROOT}/runtimes/runtimes-bins/openmp/runtime/src}"
CMAKE_BIN_DIR="${CMAKE_BIN_DIR:-/home/rodrigo.freitas/.local/lib/python3.9/site-packages/cmake/data/bin}"
PROXY_BIN="${PROXY_BIN:-}"
if [[ -z "${PROXY_BIN}" ]]; then
  if [[ -n "${BUILD_ROOT_REMOTE:-}" && -x "${BUILD_ROOT_REMOTE}/bin/llvm-offload-mpi-proxy-device" ]]; then
    PROXY_BIN="${BUILD_ROOT_REMOTE}/bin/llvm-offload-mpi-proxy-device"
  else
    PROXY_BIN="${LLVM_ROOT}/bin/llvm-offload-mpi-proxy-device"
  fi
fi

COMPARE_MODES_RAW="${HACC_COMPARE_MODES:-posix,mpiio,ompfile-wt,ompfile-wb}"
COMPARE_PARTICLES="${HACC_COMPARE_PARTICLES:-2000000}"
COMPARE_TASKS="${HACC_COMPARE_TASKS:-${SLURM_NTASKS}}"
COMPARE_OUTDIR="${HACC_COMPARE_OUTDIR:-${HACC_DIR}/tmp/backend-compare-slurm-${SLURM_JOB_ID}}"
COMPARE_CLEAN_DATA="${HACC_COMPARE_CLEAN_DATA:-1}"
# Concurrent-issue lanes: issuer threads over logical ranks (default = one
# issuer per logical rank, matching the SPMD baselines' concurrency).
COMPARE_CONCURRENCY="${HACC_COMPARE_CONCURRENCY:-${COMPARE_TASKS}}"
# srun runs 1 task/node; when the sweep uses fewer tasks than the allocation the
# node count must follow COMPARE_TASKS or srun rejects the step.
RUN_NODES="${COMPARE_TASKS}"
BUILD_HACC="${HACC_BUILD:-1}"
OMPFILE_SCHEDULER="${HACC_OMPFILE_SCHEDULER:-${LIBOMPFILE_SCHEDULER:-HEADNODE}}"

if [[ ! "${COMPARE_PARTICLES}" =~ ^[0-9]+$ ]] || (( COMPARE_PARTICLES < 1 )); then
  echo "Error: HACC_COMPARE_PARTICLES must be a positive integer, got '${COMPARE_PARTICLES}'." >&2
  exit 1
fi
if (( COMPARE_TASKS < 2 )); then
  echo "Error: HACC compare requires at least 2 tasks (split-role needs 1 app + >=1 proxy)." >&2
  exit 1
fi
if [[ ! -x "${APPTAINER_BIN}" ]]; then
  echo "Error: apptainer binary not found: ${APPTAINER_BIN}" >&2
  exit 1
fi
if [[ ! -x "${PROXY_BIN}" ]]; then
  echo "Error: proxy binary not found: ${PROXY_BIN}" >&2
  exit 1
fi

COMMON_ENV="source ${REPO_ROOT}/sh-scripts/set_env.sh ${LLVM_ROOT} && \
export REPO_ROOT=${REPO_ROOT} && \
export LLVM_ROOT=${LLVM_ROOT} && \
export LLVM_BUILD_DIR=${LLVM_ROOT} && \
export LLVM_INSTALL_ROOT=${LLVM_INSTALL_ROOT} && \
export PATH=${CMAKE_BIN_DIR}:/usr/local/mpich/bin:\${PATH} && \
export LD_LIBRARY_PATH=/usr/local/mpich/lib:${OMP_RUNTIME_LIB}:${OMPFILE_LIB}:${LLVM_ROOT}/lib:${LLVM_ROOT}/lib/x86_64-unknown-linux-gnu:${LLVM_INSTALL_ROOT}/lib:${LLVM_INSTALL_ROOT}/lib/x86_64-unknown-linux-gnu:\${LD_LIBRARY_PATH:-}"

APPTAINER_BIND_PATHS="${REPO_ROOT},/tmp,/dev/shm"
if [[ -d /scratch ]]; then
  APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS},/scratch"
fi
if [[ -d /local ]]; then
  APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS},/local"
fi

BASELINE_BIN="${HACC_DIR}/build/testhacc_io"
OMPFILE_BIN="${HACC_DIR}/build/testhacc_ompfile"
TARGET_BIN="${HACC_DIR}/build/testhacc_target"
IOR_TARGET_BIN="${HACC_DIR}/build/testior_target"

if [[ "${BUILD_HACC}" == "1" || ! -x "${BASELINE_BIN}" || ! -x "${OMPFILE_BIN}" || ! -x "${TARGET_BIN}" || ! -x "${IOR_TARGET_BIN}" ]]; then
  echo "[hacc-io-cmp] building hacc-io-fork (container cmake)"
  "${APPTAINER_BIN}" exec --userns --no-privs --bind "${APPTAINER_BIND_PATHS}" "${SIF_IMAGE}" \
    bash -lc "${COMMON_ENV} && rm -rf ${HACC_DIR}/build && cmake -S ${HACC_DIR} -B ${HACC_DIR}/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ && cmake --build ${HACC_DIR}/build -j"
fi
if [[ ! -x "${BASELINE_BIN}" || ! -x "${OMPFILE_BIN}" || ! -x "${TARGET_BIN}" || ! -x "${IOR_TARGET_BIN}" ]]; then
  echo "Error: hacc-io binaries missing after build (${BASELINE_BIN}, ${OMPFILE_BIN}, ${TARGET_BIN}, ${IOR_TARGET_BIN})." >&2
  exit 1
fi

mkdir -p "${COMPARE_OUTDIR}"
# Optional binds (/scratch, /local) exist on some nodes only — sorgan-cpu6
# has no /local, and a driver-side [[ -d ]] bakes the driver's mounts into
# every rank's bind list (job 4186: FATAL mount /local doesn't exist on
# cpu6). This wrapper decides the optional binds on the executing node.
APPTAINER_NODE_EXEC="${COMPARE_OUTDIR}/apptainer-node-exec.sh"
cat > "${APPTAINER_NODE_EXEC}" <<NODE_EXEC
#!/usr/bin/env bash
set -euo pipefail
binds="${REPO_ROOT},/tmp,/dev/shm"
[[ -d /scratch ]] && binds="\${binds},/scratch"
[[ -d /local ]] && binds="\${binds},/local"
exec "${APPTAINER_BIN}" exec --userns --no-privs --bind "\${binds}" "${SIF_IMAGE}" "\$@"
NODE_EXEC
chmod +x "${APPTAINER_NODE_EXEC}"
SUMMARY_CSV="${COMPARE_OUTDIR}/hacc-io-summary.csv"
# Columns 1-10 are positional for the SPINNER_HACC_IO awk below; the
# write-phase device-side breakdown is appended after them so that stays true.
echo "mode,interface,logical_ranks,particles_per_rank,bytes,write_s,write_mib_s,read_s,read_mib_s,verified,fill_s,issue_s,flush_s,region_s,pipeline,fill_order,pipeline_chunks,slices,launch_s" > "${SUMMARY_CSV}"
IOR_TARGET_CSV="${COMPARE_OUTDIR}/ior-target-summary.csv"
echo "mode,writers,devices,transfer_bytes,block_bytes,total_bytes,shared,read_shift,write_open_s,write_s,write_close_s,write_mib_s,read_open_s,read_s,read_close_s,read_mib_s,verified" > "${IOR_TARGET_CSV}"
# ior-target geometry (IOR -F equivalents; block matches the sorgan IOR lane)
IORT_TRANSFER_BYTES="${IORT_TRANSFER_BYTES:-1048576}"
IORT_BLOCK_BYTES="${IORT_BLOCK_BYTES:-16777216}"

echo "[hacc-io-cmp] allocated nodes: ${SLURM_JOB_NODELIST}"
echo "[hacc-io-cmp] particles/rank=${COMPARE_PARTICLES} tasks=${COMPARE_TASKS} modes=${COMPARE_MODES_RAW}"

collect_summary() { # log_file mode
  local line
  line="$(grep -ao 'HACC_IO_SUMMARY .*' "$1" | tail -1)"
  if [[ -z "${line}" ]]; then
    echo "Error: no HACC_IO_SUMMARY in $1" >&2
    tail -n 40 "$1" >&2
    exit 1
  fi
  local get
  get() { echo "${line}" | grep -oE "$1=[^ ]+" | head -1 | cut -d= -f2; }
  echo "$2,$(get interface),$(get logical_ranks),$(get particles_per_rank),$(get bytes),$(get write_s),$(get write_mib_s),$(get read_s),$(get read_mib_s),$(get verified),$(get fill_s),$(get issue_s),$(get flush_s),$(get region_s),$(get pipeline),$(get fill_order),$(get pipeline_chunks),$(get slices),$(get launch_s)" >> "${SUMMARY_CSV}"
}

run_baseline() { # mode (posix|mpiio)
  local mode="$1"
  local ifname="posix-pwrite"
  [[ "${mode}" == "mpiio" ]] && ifname="mpiio"
  local log_file="${COMPARE_OUTDIR}/${mode}.log"
  local data_path="${COMPARE_OUTDIR}/data-${mode}"
  echo "[hacc-io-cmp] baseline ${mode} (${COMPARE_TASKS} SPMD ranks)"
  srun --kill-on-bad-exit=1 --mpi=pmi2 --nodes="${RUN_NODES}" --ntasks="${COMPARE_TASKS}" \
    "${APPTAINER_NODE_EXEC}" \
    bash -lc "${COMMON_ENV} && unset LIBOMPFILE_BACKEND LIBOMPFILE_SCHEDULER LIBOMPFILE_MPP_OPEN LIBOMPFILE_MPP_IO LIBOMPFILE_MPP_PING LIBOMPFILE_OPT_OPEN_CACHE LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN LIBOMPFILE_OPT_STATS UCX_TLS UCX_POSIX_USE_PROC_LINK; export HACC_IO_INTERFACE=${ifname}; export HACC_FSYNC_ON_CLOSE=${HACC_BASELINE_FSYNC:-0}; ${BASELINE_BIN} ${COMPARE_PARTICLES} ${data_path}" \
    > "${log_file}" 2>&1 || { echo "Error: baseline ${mode} failed" >&2; tail -n 40 "${log_file}" >&2; exit 1; }
  grep -q 'CONTENTS VERIFIED' "${log_file}" || { echo "Error: baseline ${mode} did not verify" >&2; tail -n 40 "${log_file}" >&2; exit 1; }
  collect_summary "${log_file}" "${mode}"
  if [[ "${COMPARE_CLEAN_DATA}" == "1" ]]; then
    rm -f "${data_path}"*
  fi
}

run_ompfile() { # mode (ompfile-wt|ompfile-wb[-conc]) extra_env_stmt [concurrency]
  local mode="$1"
  local extra_env="$2"
  local conc="${3:-1}"
  local log_file="${COMPARE_OUTDIR}/${mode}.log"
  local data_path="${COMPARE_OUTDIR}/data-${mode}"
  echo "[hacc-io-cmp] ${mode} split-role (1 app + $((COMPARE_TASKS - 1)) proxies, logical_ranks=${COMPARE_TASKS}, concurrency=${conc})"
  srun --kill-on-bad-exit=1 --mpi=pmi2 --nodes="${RUN_NODES}" --ntasks="${COMPARE_TASKS}" \
    "${APPTAINER_NODE_EXEC}" \
    bash -lc "${COMMON_ENV} && \
      export OMP_TARGET_OFFLOAD=MANDATORY && \
      export OMPTARGET_DISABLE_HOST_PLUGIN=1 && \
      export LIBOMPFILE_BACKEND=MPI && \
      export LIBOMPFILE_MPP_OPEN=1 && \
      export LIBOMPFILE_MPP_IO=1 && \
      export LIBOMPFILE_MPP_PING=1 && \
      export LIBOMPFILE_SCHEDULER=${OMPFILE_SCHEDULER} && \
      export LIBOMPFILE_OPT_OPEN_CACHE=1 && \
      export LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN=1 && \
      export LIBOMPFILE_OPT_STATS=1 && \
      export UCX_TLS=\${UCX_TLS:-tcp,self} && \
      export UCX_POSIX_USE_PROC_LINK=\${UCX_POSIX_USE_PROC_LINK:-n} && \
      ${extra_env} \
      host_name=\$(hostname -s); \
      if [ \"\${SLURM_PROCID}\" -eq $((COMPARE_TASKS - 1)) ]; then \
        echo \"[hacc-io-cmp][rank-map] rank=\${SLURM_PROCID} role=app host=\${host_name}\"; \
        export HACC_FILE=${data_path} HACC_RANKS=${COMPARE_TASKS} HACC_PARTICLES=${COMPARE_PARTICLES} HACC_CONCURRENCY=${conc}; \
        exec ${OMPFILE_BIN}; \
      else \
        echo \"[hacc-io-cmp][rank-map] rank=\${SLURM_PROCID} role=proxy host=\${host_name}\"; \
        exec ${PROXY_BIN}; \
      fi" \
    > "${log_file}" 2>&1 || { echo "Error: ${mode} failed" >&2; tail -n 60 "${log_file}" >&2; exit 1; }
  grep -q 'CONTENTS VERIFIED' "${log_file}" || { echo "Error: ${mode} did not verify" >&2; tail -n 60 "${log_file}" >&2; exit 1; }
  collect_summary "${log_file}" "${mode}"
  if [[ "${COMPARE_CLEAN_DATA}" == "1" ]]; then
    rm -f "${data_path}"*
  fi
}

run_ompfile_target() { # mode extra_env [fpr] [async] [segmented] [concurrent] [pipeline] [pipeline_flush] [fill_order] [slices]
  local mode="$1"
  local extra_env="$2"
  local fpr="${3:-0}"
  local async="${4:-0}"
  local segmented="${5:-0}"
  # Segment-shape sweep, matching the AMD wrapper: HACC_SEG_COUNT=N overrides
  # the segment count for the segmented modes only (N>1 = equal chunks in the
  # driver; 1 = legacy 9-way GLEAN split).
  if [[ "${segmented}" == "1" && -n "${HACC_SEG_COUNT:-}" ]]; then
    segmented="${HACC_SEG_COUNT}"
  fi
  local concurrent="${6:-0}"
  # Fill/write pipelining: the region generates chunk c and hands it to the
  # async engine before generating chunk c+1, instead of filling the whole rank
  # block first. Only meaningful with async=1 and a segmented shape.
  local pipeline="${7:-0}"
  # Per-mode so a flush and a no-flush pipelined arm can share one job; the
  # env var stays the global override for modes that do not pass an argument.
  local pipeline_flush="${8:-${HACC_PIPELINE_FLUSH:-1}}"
  # particle (default) or array: the buffer-fill traversal order for the
  # non-pipelined path. `array` is 1.63x slower (job 414888) and exists only as
  # the control that measured that; the pipeline always fills particle-major.
  local fill_order="${9:-${HACC_FILL_ORDER:-particle}}"
  # Particle slices per logical rank dealt round-robin over devices with
  # per-device dependence queues (1 = whole-rank waves). "auto" = one slice per
  # proxy. Only the segmented shared-file modes may set it.
  local slices="${10:-1}"
  if [[ "${slices}" == "auto" ]]; then slices=$((COMPARE_TASKS - 1)); fi
  # Sliced regions overlap through the async queue, which must hold two
  # regions' worth of writes (9 each) rather than the runtime default of 2.
  # Same "export ... &&" shape as the extra_env blocks it is appended to.
  local slice_env=""
  if (( slices > 1 )); then
    slice_env="export LIBOMPFILE_ASYNC_QUEUE_DEPTH=\${LIBOMPFILE_ASYNC_QUEUE_DEPTH:-${HACC_SLICE_QUEUE_DEPTH:-18}} &&"
  fi
  local log_file="${COMPARE_OUTDIR}/${mode}.log"
  local data_path="${COMPARE_OUTDIR}/data-${mode}"
  echo "[hacc-io-cmp] ${mode} target-region issuers (1 app + $((COMPARE_TASKS - 1)) proxies, logical_ranks=${COMPARE_TASKS}, data generated device-side)"
  srun --kill-on-bad-exit=1 --mpi=pmi2 --nodes="${RUN_NODES}" --ntasks="${COMPARE_TASKS}" \
    "${APPTAINER_NODE_EXEC}" \
    bash -lc "${COMMON_ENV} && \
      export OMP_TARGET_OFFLOAD=MANDATORY && \
      export OMPTARGET_DISABLE_HOST_PLUGIN=1 && \
      export LIBOMPFILE_BACKEND=MPI && \
      export LIBOMPFILE_MPP_OPEN=1 && \
      export LIBOMPFILE_MPP_IO=1 && \
      export LIBOMPFILE_MPP_PING=1 && \
      export LIBOMPFILE_SCHEDULER=${OMPFILE_SCHEDULER} && \
      export LIBOMPFILE_OPT_OPEN_CACHE=1 && \
      export LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN=1 && \
      export LIBOMPFILE_OPT_STATS=1 && \
      export UCX_TLS=\${UCX_TLS:-tcp,self} && \
      export UCX_POSIX_USE_PROC_LINK=\${UCX_POSIX_USE_PROC_LINK:-n} && \
      ${extra_env} \
      ${slice_env} \
      host_name=\$(hostname -s); \
      if [ \"\${SLURM_PROCID}\" -eq $((COMPARE_TASKS - 1)) ]; then \
        echo \"[hacc-io-cmp][rank-map] rank=\${SLURM_PROCID} role=app host=\${host_name}\"; \
        export HACC_FILE=${data_path} HACC_RANKS=${COMPARE_TASKS} HACC_PARTICLES=${COMPARE_PARTICLES} HACC_FILE_PER_RANK=${fpr} HACC_IO_ASYNC=${async} HACC_SEGMENTED=${segmented} HACC_CONCURRENT_OPEN=${concurrent} HACC_PIPELINE=${pipeline} HACC_PIPELINE_FLUSH=${pipeline_flush} HACC_FILL_ORDER=${fill_order} HACC_PIPELINE_CHUNKS=${HACC_PIPELINE_CHUNKS:-4} HACC_SLICES=${slices}; \
        exec ${TARGET_BIN}; \
      else \
        echo \"[hacc-io-cmp][rank-map] rank=\${SLURM_PROCID} role=proxy host=\${host_name}\"; \
        exec ${PROXY_BIN}; \
      fi" \
    > "${log_file}" 2>&1 || { echo "Error: ${mode} failed" >&2; tail -n 60 "${log_file}" >&2; exit 1; }
  grep -q 'CONTENTS VERIFIED' "${log_file}" || { echo "Error: ${mode} did not verify" >&2; tail -n 60 "${log_file}" >&2; exit 1; }
  collect_summary "${log_file}" "${mode}"
  if [[ "${COMPARE_CLEAN_DATA}" == "1" ]]; then
    rm -f "${data_path}"*
  fi
}

collect_ior_target_summary() { # log_file mode
  local line
  line="$(grep -ao 'IOR_TARGET_SUMMARY .*' "$1" | tail -1)"
  if [[ -z "${line}" ]]; then
    echo "Error: no IOR_TARGET_SUMMARY in $1" >&2
    tail -n 40 "$1" >&2
    exit 1
  fi
  local get
  get() { echo "${line}" | grep -oE "$1=[^ ]+" | head -1 | cut -d= -f2; }
  echo "$2,$(get writers),$(get devices),$(get transfer_bytes),$(get block_bytes),$(get total_bytes),$(get shared),$(get read_shift),$(get write_open_s),$(get write_s),$(get write_close_s),$(get write_mib_s),$(get read_open_s),$(get read_s),$(get read_close_s),$(get read_mib_s),$(get verified)" >> "${IOR_TARGET_CSV}"
}

run_ior_target() { # mode extra_env_stmt [shared] [read_shift]
  local mode="$1"
  local extra_env="$2"
  local shared="${3:-0}"
  local read_shift="${4:-0}"
  local log_file="${COMPARE_OUTDIR}/${mode}.log"
  local data_path="${COMPARE_OUTDIR}/data-${mode}"
  echo "[hacc-io-cmp] ${mode} target-region IOR pattern (writers=proxies, transfer=${IORT_TRANSFER_BYTES} block=${IORT_BLOCK_BYTES} shared=${shared} read_shift=${read_shift})"
  srun --kill-on-bad-exit=1 --mpi=pmi2 --nodes="${RUN_NODES}" --ntasks="${COMPARE_TASKS}" \
    "${APPTAINER_NODE_EXEC}" \
    bash -lc "${COMMON_ENV} && \
      export OMP_TARGET_OFFLOAD=MANDATORY && \
      export OMPTARGET_DISABLE_HOST_PLUGIN=1 && \
      export LIBOMPFILE_BACKEND=MPI && \
      export LIBOMPFILE_MPP_OPEN=1 && \
      export LIBOMPFILE_MPP_IO=1 && \
      export LIBOMPFILE_MPP_PING=1 && \
      export LIBOMPFILE_SCHEDULER=${OMPFILE_SCHEDULER} && \
      export LIBOMPFILE_OPT_OPEN_CACHE=1 && \
      export LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN=1 && \
      export LIBOMPFILE_OPT_STATS=1 && \
      export UCX_TLS=\${UCX_TLS:-tcp,self} && \
      export UCX_POSIX_USE_PROC_LINK=\${UCX_POSIX_USE_PROC_LINK:-n} && \
      ${extra_env} \
      host_name=\$(hostname -s); \
      if [ \"\${SLURM_PROCID}\" -eq $((COMPARE_TASKS - 1)) ]; then \
        echo \"[hacc-io-cmp][rank-map] rank=\${SLURM_PROCID} role=app host=\${host_name}\"; \
        export IORT_FILE=${data_path} IORT_TRANSFER_BYTES=${IORT_TRANSFER_BYTES} IORT_BLOCK_BYTES=${IORT_BLOCK_BYTES}; \
        export IORT_SHARED=${shared} IORT_READ_SHIFT=${read_shift}; \
        exec ${IOR_TARGET_BIN}; \
      else \
        echo \"[hacc-io-cmp][rank-map] rank=\${SLURM_PROCID} role=proxy host=\${host_name}\"; \
        exec ${PROXY_BIN}; \
      fi" \
    > "${log_file}" 2>&1 || { echo "Error: ${mode} failed" >&2; tail -n 60 "${log_file}" >&2; exit 1; }
  grep -q 'CONTENTS VERIFIED' "${log_file}" || { echo "Error: ${mode} did not verify" >&2; tail -n 60 "${log_file}" >&2; exit 1; }
  collect_ior_target_summary "${log_file}" "${mode}"
  if [[ "${COMPARE_CLEAN_DATA}" == "1" ]]; then
    rm -f "${data_path}"*
  fi
}

# Topology for the write-back lane: node-local stage dirs on the proxy hosts,
# authoritative storage on the shared filesystem (same shape as the OOC lane).
TOPOLOGY_FILE="${COMPARE_OUTDIR}/topology.txt"
{
  if [[ -d /work1 ]]; then
    echo "global /work1 parallel_fs"
  else
    echo "global /scratch nfs"
  fi
  scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n $((COMPARE_TASKS - 1)) | while read -r host; do
    echo "host ${host} /tmp/${USER}/libompfile-stage/hacc-${SLURM_JOB_ID} local-posix"
  done
} > "${TOPOLOGY_FILE}"

WB_ENV="export LIBOMPFILE_STAGE_MODE=readthrough && \
export LIBOMPFILE_STAGE_ROOT_POLICY=topology && \
export LIBOMPFILE_TOPOLOGY_FILE=${TOPOLOGY_FILE} && \
export LIBOMPFILE_STAGE_WRITE_MODE=write-back && \
export LIBOMPFILE_STAGE_DIRTY_WATERMARK_BYTES=8388608 && \
export LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY=close &&"
WT_ENV="export LIBOMPFILE_STAGE_MODE=off && \
export LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY=each &&"
# Fair-posture write-through: baselines never fsync per write, so the
# target-region lanes default to the close policy (override via env).
TARGET_WT_ENV="export LIBOMPFILE_STAGE_MODE=off && \
export LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY=${HACC_TARGET_FSYNC_POLICY:-close} &&"
# Owner-bypass: shared file, but each proxy reads/writes its disjoint range
# locally instead of forwarding to the file owner. The read flag is required
# too: without it restart preads funnel through the owner's single stream
# (see the AMD variant's comment; AMD segsweep job 372096 showed
# local_disjoint_reads=0 with reads flat at 1.4-1.7 GiB/s vs 5.6-8.5 writes).
TARGET_LOCAL_BATCHED_ENV="export LIBOMPFILE_STAGE_MODE=off && \
export LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY=${HACC_TARGET_FSYNC_POLICY:-close} && \
export LIBOMPFILE_OPT_LOCAL_DISJOINT_WRITE=1 && \
export LIBOMPFILE_OPT_LOCAL_DISJOINT_READ=1 &&"
# DEFAULT since Sep 10, 2026 — see the AMD lane for the rationale and numbers.
# Worth 1.20x write / 1.74x read here on NFS (job 429) and 1.69x median write
# on wekafs, so the window removal is not a fast-storage tweak.
# TARGET_LOCAL_BATCHED_ENV keeps the pre-Sep-10 behaviour for older rows.
TARGET_LOCAL_ENV="${TARGET_LOCAL_BATCHED_ENV} \
export LIBOMPFILE_OPT_WRITE_BATCH=disabled && \
export LIBOMPFILE_OPT_TWO_PHASE=disabled && \
export LIBOMPFILE_OPT_LOCAL_DISJOINT_EXPRESS=1 &&"
# Concurrent-open validation lane bypasses the two-phase batch read planner to
# dodge an unrelated intermittent MPI_Testany truncation in that path (see
# docs/known-issues.md); scoped to this lane so the campaign two-phase story stays.
TARGET_LOCAL_NOTWOPHASE_ENV="${TARGET_LOCAL_BATCHED_ENV} export LIBOMPFILE_OPT_TWO_PHASE=disabled &&"
# Striped bypass I/O (see the AMD wrapper): K concurrent stripes per large
# bypass op on the proxy. A control here — NFS is expected to lose.
TARGET_LOCAL_STREAMS_ENV="${TARGET_LOCAL_ENV} \
export LIBOMPFILE_OPT_LOCAL_DISJOINT_WRITE_STREAMS=${HACC_WRITE_STREAMS:-4} && \
export LIBOMPFILE_OPT_LOCAL_DISJOINT_READ_STREAMS=${HACC_READ_STREAMS:-4} &&"
# Owner-bypass on BOTH directions: alias — TARGET_LOCAL_ENV now carries the
# read bypass itself.
TARGET_LOCAL_RW_ENV="${TARGET_LOCAL_ENV}"
# No-batch bypass: the write-batch and two-phase planners each hold every op in
# a 2 ms grouping window to merge funnel traffic across ranks. With one writer
# per proxy and the bypass serving ops locally nobody joins the batch
# (write_queue_max_depth=1), so the window is a pure per-op tax. Mirrors the
# AMD lane's block of the same name.
TARGET_LOCAL_RW_NB_ENV="${TARGET_LOCAL_RW_ENV}
export LIBOMPFILE_OPT_WRITE_BATCH=disabled
export LIBOMPFILE_OPT_TWO_PHASE=disabled
export LIBOMPFILE_OPT_LOCAL_DISJOINT_EXPRESS=1"
# Coalesced bypass: no-batch plus the proxy-side op-coalescing layer.
TARGET_LOCAL_RW_CO_ENV="${TARGET_LOCAL_RW_NB_ENV}
export LIBOMPFILE_OPT_LOCAL_DISJOINT_WRITE_COMBINE=${HACC_TARGET_COALESCE_BYTES:-16777216}
export LIBOMPFILE_OPT_LOCAL_DISJOINT_READAHEAD=${HACC_TARGET_COALESCE_BYTES:-16777216}"

MODE_LIST="${COMPARE_MODES_RAW//,/ }"
for mode in ${MODE_LIST}; do
  case "${mode}" in
    posix|mpiio) run_baseline "${mode}" ;;
    ompfile-wt) run_ompfile ompfile-wt "${WT_ENV}" 1 ;;
    ompfile-wb) run_ompfile ompfile-wb "${WB_ENV}" 1 ;;
    ompfile-wt-conc) run_ompfile ompfile-wt-conc "${WT_ENV}" "${COMPARE_CONCURRENCY}" ;;
    ompfile-wb-conc) run_ompfile ompfile-wb-conc "${WB_ENV}" "${COMPARE_CONCURRENCY}" ;;
    ompfile-target-wt) run_ompfile_target ompfile-target-wt "${TARGET_WT_ENV}" 0 ;;
    ompfile-target-local) run_ompfile_target ompfile-target-local "${TARGET_LOCAL_ENV}" 0 0 0 ;;
    ompfile-target-local-async) run_ompfile_target ompfile-target-local-async "${TARGET_LOCAL_ENV}" 0 1 0 ;;
    ompfile-target-local-seg) run_ompfile_target ompfile-target-local-seg "${TARGET_LOCAL_ENV}" 0 0 1 ;;
    ompfile-target-local-seg-async) run_ompfile_target ompfile-target-local-seg-async "${TARGET_LOCAL_ENV}" 0 1 1 0 0 ;;
    # Identical to the line above except the buffer fill walks one GLEAN
    # segment at a time instead of one pass over particles. Kept as the control
    # that measured the fill traversal to be worth 1.23x of the write phase
    # (job 414888); it is the slower order, not the default.
    ompfile-target-local-seg-async-amfill) run_ompfile_target ompfile-target-local-seg-async-amfill "${TARGET_LOCAL_ENV}" 0 1 1 0 0 1 array ;;
    # Same shape, fill/write pipelined: each proxy generates particle slice
    # c+1 while the async engine drains slice c. A/B partner of the line above
    # — same env, same particle-major fill, same bytes; sweep the slice count
    # with HACC_PIPELINE_CHUNKS (default 4), not HACC_SEG_COUNT.
    ompfile-target-local-seg-async-pipe) run_ompfile_target ompfile-target-local-seg-async-pipe "${TARGET_LOCAL_ENV}" 0 1 1 0 1 1 ;;
    # Pipelined without the per-region omp_file_flush: the queued writes drain
    # at close instead. Ablation partner of the line above.
    ompfile-target-local-seg-async-pipe-noflush) run_ompfile_target ompfile-target-local-seg-async-pipe-noflush "${TARGET_LOCAL_ENV}" 0 1 1 0 1 0 ;;
    # Sliced dispatch: one slice per proxy per logical rank, dealt round-robin
    # over devices through per-device dependence queues (4 ranks on 3 proxies
    # becomes 4/4/4 regions instead of 2/1/1 blocks). HACC_SLICES=N overrides.
    ompfile-target-local-seg-async-slice) run_ompfile_target ompfile-target-local-seg-async-slice "${TARGET_LOCAL_ENV}" 0 1 1 0 0 1 particle "${HACC_SLICES:-auto}" ;;
    ompfile-target-local-seg-async-slice-pipe) run_ompfile_target ompfile-target-local-seg-async-slice-pipe "${TARGET_LOCAL_ENV}" 0 1 1 0 1 0 particle "${HACC_SLICES:-auto}" ;;
    # Sliced dispatch plus striped bypass I/O on the proxy: the sliced arm's
    # env with LIBOMPFILE_OPT_LOCAL_DISJOINT_{WRITE,READ}_STREAMS (default 4,
    # HACC_WRITE_STREAMS / HACC_READ_STREAMS override). Its A/B partner is the
    # -slice line: same dispatch, same ops, only the proxy's drain differs.
    ompfile-target-local-seg-async-slice-streams) run_ompfile_target ompfile-target-local-seg-async-slice-streams "${TARGET_LOCAL_STREAMS_ENV}" 0 1 1 0 0 1 particle "${HACC_SLICES:-auto}" ;;
    # Same shape with the client-side grouping windows off and the express
    # path on. The bypass serves ops locally, so nobody ever joins the 2 ms
    # write-batch/two-phase windows and they are a pure per-op tax
    # (write_queue_max_depth=1 in every HACC run measured so far). These two
    # cases carry the env the IOR lanes already use; HACC had never been run
    # with it.
    ompfile-target-local-nb-seg-async) run_ompfile_target ompfile-target-local-nb-seg-async "${TARGET_LOCAL_RW_NB_ENV}" 0 1 1 ;;
    # Coalesced (one pwrite per region instead of the nine GLEAN segments):
    # isolates per-op dispatch cost from per-byte cost at a fixed proxy count.
    ompfile-target-local-nb-async) run_ompfile_target ompfile-target-local-nb-async "${TARGET_LOCAL_RW_NB_ENV}" 0 1 0 ;;
    ompfile-target-local-batched-seg-async) run_ompfile_target ompfile-target-local-batched-seg-async "${TARGET_LOCAL_BATCHED_ENV}" 0 1 1 ;;
    ompfile-target-local-co-seg-async) run_ompfile_target ompfile-target-local-co-seg-async "${TARGET_LOCAL_RW_CO_ENV}" 0 1 1 ;;
    ompfile-target-wb) run_ompfile_target ompfile-target-wb "${WB_ENV}" 0 ;;
    ompfile-target-fpr) run_ompfile_target ompfile-target-fpr "${TARGET_WT_ENV}" 1 ;;
    ompfile-target-fpr-concurrent) run_ompfile_target ompfile-target-fpr-concurrent "${TARGET_LOCAL_ENV}" 1 0 0 1 ;;
    ior-target) run_ior_target ior-target "${TARGET_WT_ENV}" ;;
    ior-target-shared) run_ior_target ior-target-shared "${TARGET_WT_ENV}" 1 0 ;;
    ior-target-shared-shift) run_ior_target ior-target-shared-shift "${TARGET_WT_ENV}" 1 1 ;;
    ior-target-shared-shift-local) run_ior_target ior-target-shared-shift-local "${TARGET_LOCAL_ENV}" 1 1 ;;
    ior-target-shared-shift-local-rw) run_ior_target ior-target-shared-shift-local-rw "${TARGET_LOCAL_RW_ENV}" 1 1 ;;
    *) echo "Error: unsupported mode '${mode}' in HACC_COMPARE_MODES." >&2; exit 1 ;;
  esac
done

echo "[hacc-io-cmp] summary:"
cat "${SUMMARY_CSV}"
if [[ "$(wc -l < "${IOR_TARGET_CSV}")" -gt 1 ]]; then
  cat "${IOR_TARGET_CSV}"
fi

# One machine-parseable line per summary for the spinner capture layer.
if [[ "$(wc -l < "${SUMMARY_CSV}")" -gt 1 ]]; then
  awk -F, 'NR > 1 {
    gsub(/-/, "_", $1);
    w[$1] = $7; r[$1] = $9; v[$1] = $10; bytes = $5; ranks = $3; particles = $4;
  }
  END {
    printf "SPINNER_HACC_IO tasks=%s particles_per_rank=%s bytes=%s", ranks, particles, bytes;
    for (m in w) printf " write_mib_%s=%s read_mib_%s=%s verified_%s=%s", m, w[m], m, r[m], m, v[m];
    printf "\n";
  }' "${SUMMARY_CSV}"
fi
if [[ "$(wc -l < "${IOR_TARGET_CSV}")" -gt 1 ]]; then
  awk -F, 'NR > 1 {
    writers = $2; devices = $3; transfer = $4; block = $5;
    wopen = $7; wio = $8; wbw = $10; rio = $12; rbw = $14; ver = $15;
  }
  END {
    printf "SPINNER_IOR_TARGET writers=%s devices=%s transfer_bytes=%s block_bytes=%s write_open_s=%s write_s=%s write_mib_s=%s read_s=%s read_mib_s=%s verified=%s\n",
      writers, devices, transfer, block, wopen, wio, wbw, rio, rbw, ver;
  }' "${IOR_TARGET_CSV}"
fi

echo "[hacc-io-cmp] status=PASS summary_csv=${SUMMARY_CSV}"
