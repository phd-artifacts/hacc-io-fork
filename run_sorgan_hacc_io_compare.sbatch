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
SUMMARY_CSV="${COMPARE_OUTDIR}/hacc-io-summary.csv"
echo "mode,interface,logical_ranks,particles_per_rank,bytes,write_s,write_mib_s,read_s,read_mib_s,verified" > "${SUMMARY_CSV}"
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
  echo "$2,$(get interface),$(get logical_ranks),$(get particles_per_rank),$(get bytes),$(get write_s),$(get write_mib_s),$(get read_s),$(get read_mib_s),$(get verified)" >> "${SUMMARY_CSV}"
}

run_baseline() { # mode (posix|mpiio)
  local mode="$1"
  local ifname="posix-pwrite"
  [[ "${mode}" == "mpiio" ]] && ifname="mpiio"
  local log_file="${COMPARE_OUTDIR}/${mode}.log"
  local data_path="${COMPARE_OUTDIR}/data-${mode}"
  echo "[hacc-io-cmp] baseline ${mode} (${COMPARE_TASKS} SPMD ranks)"
  srun --kill-on-bad-exit=1 --mpi=pmi2 --nodes="${RUN_NODES}" --ntasks="${COMPARE_TASKS}" \
    "${APPTAINER_BIN}" exec --userns --no-privs --bind "${APPTAINER_BIND_PATHS}" "${SIF_IMAGE}" \
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
    "${APPTAINER_BIN}" exec --userns --no-privs --bind "${APPTAINER_BIND_PATHS}" "${SIF_IMAGE}" \
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

run_ompfile_target() { # mode extra_env_stmt [file_per_rank] [async] [segmented]
  local mode="$1"
  local extra_env="$2"
  local fpr="${3:-0}"
  local async="${4:-0}"
  local segmented="${5:-0}"
  local concurrent="${6:-0}"
  local log_file="${COMPARE_OUTDIR}/${mode}.log"
  local data_path="${COMPARE_OUTDIR}/data-${mode}"
  echo "[hacc-io-cmp] ${mode} target-region issuers (1 app + $((COMPARE_TASKS - 1)) proxies, logical_ranks=${COMPARE_TASKS}, data generated device-side)"
  srun --kill-on-bad-exit=1 --mpi=pmi2 --nodes="${RUN_NODES}" --ntasks="${COMPARE_TASKS}" \
    "${APPTAINER_BIN}" exec --userns --no-privs --bind "${APPTAINER_BIND_PATHS}" "${SIF_IMAGE}" \
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
        export HACC_FILE=${data_path} HACC_RANKS=${COMPARE_TASKS} HACC_PARTICLES=${COMPARE_PARTICLES} HACC_FILE_PER_RANK=${fpr} HACC_IO_ASYNC=${async} HACC_SEGMENTED=${segmented} HACC_CONCURRENT_OPEN=${concurrent}; \
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
    "${APPTAINER_BIN}" exec --userns --no-privs --bind "${APPTAINER_BIND_PATHS}" "${SIF_IMAGE}" \
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
# Owner-bypass: shared file, but each proxy writes its disjoint range locally
# (LIBOMPFILE_OPT_LOCAL_DISJOINT_WRITE=1) instead of forwarding to the file owner.
TARGET_LOCAL_ENV="export LIBOMPFILE_STAGE_MODE=off && \
export LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY=${HACC_TARGET_FSYNC_POLICY:-close} && \
export LIBOMPFILE_OPT_LOCAL_DISJOINT_WRITE=1 &&"
# Concurrent-open validation lane bypasses the two-phase batch read planner to
# dodge an unrelated intermittent MPI_Testany truncation in that path (see
# docs/known-issues.md); scoped to this lane so the campaign two-phase story stays.
TARGET_LOCAL_NOTWOPHASE_ENV="${TARGET_LOCAL_ENV} export LIBOMPFILE_OPT_TWO_PHASE=disabled &&"
# Owner-bypass on BOTH directions (see AMD variant).
TARGET_LOCAL_RW_ENV="${TARGET_LOCAL_ENV} && export LIBOMPFILE_OPT_LOCAL_DISJOINT_READ=1"

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
    ompfile-target-local-seg-async) run_ompfile_target ompfile-target-local-seg-async "${TARGET_LOCAL_ENV}" 0 1 1 ;;
    ompfile-target-wb) run_ompfile_target ompfile-target-wb "${WB_ENV}" 0 ;;
    ompfile-target-fpr) run_ompfile_target ompfile-target-fpr "${TARGET_WT_ENV}" 1 ;;
    ompfile-target-fpr-concurrent) run_ompfile_target ompfile-target-fpr-concurrent "${TARGET_LOCAL_NOTWOPHASE_ENV}" 1 0 0 1 ;;
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
