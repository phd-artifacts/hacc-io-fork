#!/bin/bash
#SBATCH --job-name=amd-hacc-io-cmp
#SBATCH --partition=mi2104x
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

# AMD HACC-IO backend compare — same lane layout as the sorgan wrapper
# (run_sorgan_hacc_io_compare.sbatch): posix/mpiio SPMD baselines vs the
# split-role OMPFILE variant in write-through (wt) and write-back+close (wb)
# postures. Launch mechanics follow the OOC AMD lane: ranks run inside the
# container via a role script, srun --mpi=pmix, host spack MPICH override
# (container MPICH's PMIx client silently singletons on this cluster —
# jobs 350109-350170), and PMIx server tmpdir bind.

set -euo pipefail

module purge 2>/dev/null || true
module load hpcfund 2>/dev/null || true

PROJECT_NAME="${PROJECT_NAME:-io-playground}"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -z "${WORK:-}" ]]; then
    echo "Error: set PROJECT_ROOT or ensure WORK is exported by the cluster environment." >&2
    exit 1
  fi
  PROJECT_ROOT="${WORK}/projects/${PROJECT_NAME}"
fi
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/src}"
HACC_DIR="${HACC_DIR:-${REPO_ROOT}/application/hacc-io-fork}"
LLVM_ROOT_REMOTE="${LLVM_ROOT_REMOTE:-${PROJECT_ROOT}/llvm-infra/llvm-builds/apptainer-Debug}"
BUILD_ROOT_REMOTE="${BUILD_ROOT_REMOTE:-${PROJECT_ROOT}/builds/latest/gfx90a}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${REPO_ROOT}/sifs/ompc-base_latest.sif}"
if [[ ! -e "${CONTAINER_IMAGE}" ]]; then
  CONTAINER_IMAGE="${REPO_ROOT}/ompc-base_latest.sif"
fi
PROXY_BIN="${PROXY_BIN:-}"
if [[ -z "${PROXY_BIN}" ]]; then
  if [[ -x "${BUILD_ROOT_REMOTE}/bin/llvm-offload-mpi-proxy-device" ]]; then
    PROXY_BIN="${BUILD_ROOT_REMOTE}/bin/llvm-offload-mpi-proxy-device"
  else
    PROXY_BIN="${LLVM_ROOT_REMOTE}/bin/llvm-offload-mpi-proxy-device"
  fi
fi

COMPARE_MODES_RAW="${HACC_COMPARE_MODES:-posix,mpiio,ompfile-wt,ompfile-wb}"
COMPARE_PARTICLES="${HACC_COMPARE_PARTICLES:-2000000}"
COMPARE_TASKS="${HACC_COMPARE_TASKS:-${SLURM_NTASKS}}"
COMPARE_OUTDIR="${HACC_COMPARE_OUTDIR:-${HACC_DIR}/tmp/backend-compare-amd-${SLURM_JOB_ID}}"
COMPARE_CLEAN_DATA="${HACC_COMPARE_CLEAN_DATA:-1}"
# Concurrent-issue lanes: issuer threads over logical ranks (default = one
# issuer per logical rank, matching the SPMD baselines' concurrency).
COMPARE_CONCURRENCY="${HACC_COMPARE_CONCURRENCY:-${COMPARE_TASKS}}"
# srun runs 1 task/node; when the sweep uses fewer tasks than the allocation the
# node count must follow COMPARE_TASKS or srun rejects the step.
RUN_NODES="${COMPARE_TASKS}"
BUILD_HACC="${HACC_BUILD:-1}"
OMPFILE_SCHEDULER="${HACC_OMPFILE_SCHEDULER:-${LIBOMPFILE_SCHEDULER:-HEADNODE}}"
RUN_MPI_TYPE="${RUN_MPI_TYPE:-pmix}"

if [[ ! "${COMPARE_PARTICLES}" =~ ^[0-9]+$ ]] || (( COMPARE_PARTICLES < 1 )); then
  echo "Error: HACC_COMPARE_PARTICLES must be a positive integer, got '${COMPARE_PARTICLES}'." >&2
  exit 1
fi
if (( COMPARE_TASKS < 2 )); then
  echo "Error: HACC compare requires at least 2 tasks." >&2
  exit 1
fi
if [[ ! -e "${CONTAINER_IMAGE}" ]]; then
  echo "Error: Apptainer image not found under ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${PROXY_BIN}" ]]; then
  echo "Error: proxy binary not found: ${PROXY_BIN}" >&2
  exit 1
fi

# Host spack MPICH (pmi=pmix, cluster-matched) — required for every rank so
# the PMIx client inside the container can bootstrap (see OOC lane notes).
SPACK_ROOT="${SPACK_ROOT:-${HOME}/spack}"
MPICH_ROOT=""
if [[ -f "${SPACK_ROOT}/share/spack/setup-env.sh" ]]; then
  # shellcheck disable=SC1091
  . "${SPACK_ROOT}/share/spack/setup-env.sh" 2>/dev/null || true
  MPICH_ROOT="$(spack location -i mpich@4.2.3 2>/dev/null || true)"
fi
if [[ ! -d "${MPICH_ROOT}" ]]; then
  echo "Error: spack mpich@4.2.3 not found; host-MPI override unavailable." >&2
  exit 1
fi
MPI_LIB_OVERRIDE="${MPICH_ROOT}/lib"

mkdir -p "${COMPARE_OUTDIR}"
SUMMARY_CSV="${COMPARE_OUTDIR}/hacc-io-summary.csv"
echo "mode,interface,logical_ranks,particles_per_rank,bytes,write_s,write_mib_s,read_s,read_mib_s,verified" > "${SUMMARY_CSV}"
IOR_TARGET_CSV="${COMPARE_OUTDIR}/ior-target-summary.csv"
echo "mode,writers,devices,transfer_bytes,block_bytes,total_bytes,write_open_s,write_s,write_close_s,write_mib_s,read_open_s,read_s,read_close_s,read_mib_s,verified" > "${IOR_TARGET_CSV}"
# ior-target geometry (IOR -F equivalents; block matches the AMD IOR lane)
IORT_TRANSFER_BYTES="${IORT_TRANSFER_BYTES:-1048576}"
IORT_BLOCK_BYTES="${IORT_BLOCK_BYTES:-134217728}"

echo "[amd-hacc-io-cmp] allocated nodes: ${SLURM_JOB_NODELIST}"
echo "[amd-hacc-io-cmp] particles/rank=${COMPARE_PARTICLES} tasks=${COMPARE_TASKS} modes=${COMPARE_MODES_RAW}"

in_container_build() {
  local binds="${PROJECT_ROOT},${REPO_ROOT},${HOME}"
  [[ -d /work1 ]] && binds="${binds},/work1"
  APPTAINER_DISABLE_CACHE=1 apptainer exec --userns --no-privs --bind "${binds}" "${CONTAINER_IMAGE}" \
    /bin/bash -lc "source '${REPO_ROOT}/sh-scripts/set_env.sh' '${LLVM_ROOT_REMOTE}' >/dev/null; export LLVM_BUILD_DIR='${LLVM_ROOT_REMOTE}' LLVM_INSTALL_DIR='${LLVM_ROOT_REMOTE}' LLVM_ROOT='${LLVM_ROOT_REMOTE}'; rm -rf '${HACC_DIR}/build' && cmake -S '${HACC_DIR}' -B '${HACC_DIR}/build' -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ && cmake --build '${HACC_DIR}/build' -j"
}

BASELINE_BIN="${HACC_DIR}/build/testhacc_io"
OMPFILE_BIN="${HACC_DIR}/build/testhacc_ompfile"
TARGET_BIN="${HACC_DIR}/build/testhacc_target"
IOR_TARGET_BIN="${HACC_DIR}/build/testior_target"

if [[ "${BUILD_HACC}" == "1" || ! -x "${BASELINE_BIN}" || ! -x "${OMPFILE_BIN}" || ! -x "${TARGET_BIN}" || ! -x "${IOR_TARGET_BIN}" ]]; then
  echo "[amd-hacc-io-cmp] building hacc-io-fork (container cmake)"
  in_container_build
fi
if [[ ! -x "${BASELINE_BIN}" || ! -x "${OMPFILE_BIN}" || ! -x "${TARGET_BIN}" || ! -x "${IOR_TARGET_BIN}" ]]; then
  echo "Error: hacc-io binaries missing after build (${BASELINE_BIN}, ${OMPFILE_BIN}, ${TARGET_BIN}, ${IOR_TARGET_BIN})." >&2
  exit 1
fi

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

# Role script: every rank re-enters the container with the PMIx server bind
# and the host-MPI lib override, then execs its target binary. Modeled on
# application/benchmarks/ooc-cholesky-io/run.sh.
write_role_script() { # path env_stmt role_case ("baseline" runs app on every rank)
  local role_path="$1" env_stmt="$2" role_case="$3"
  cat > "${role_path}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
${env_stmt}

run_in_container() {
  local target_bin="\$1"
  local binds="${PROJECT_ROOT}:${PROJECT_ROOT},${REPO_ROOT}:${REPO_ROOT},${COMPARE_OUTDIR}:${COMPARE_OUTDIR},/tmp:/tmp,/dev/shm:/dev/shm"
  if [[ -d /work1 ]]; then binds="\${binds},/work1:/work1"; fi
  if [[ -n "\${PMIX_SERVER_TMPDIR:-}" ]]; then
    pmix_dir="\$(dirname "\${PMIX_SERVER_TMPDIR}")"
    if [[ -d "\${pmix_dir}" ]]; then binds="\${binds},\${pmix_dir}:\${pmix_dir}"; fi
  elif [[ -d /var/spool/slurmd ]]; then
    binds="\${binds},/var/spool/slurmd:/var/spool/slurmd"
  fi
  if [[ -d "\${HOME}" ]]; then binds="\${binds},\${HOME}:\${HOME}"; fi
  if [[ -d /opt ]]; then binds="\${binds},/opt:/opt"; fi
  APPTAINER_DISABLE_CACHE=1 exec apptainer exec \\
    --userns --no-privs \\
    --bind "\${binds}" \\
    "${CONTAINER_IMAGE}" \\
    /bin/bash -lc "source \"${REPO_ROOT}/sh-scripts/set_env.sh\" \"${LLVM_ROOT_REMOTE}\" >/dev/null && export LLVM_BUILD_DIR=\"${LLVM_ROOT_REMOTE}\" LLVM_INSTALL_DIR=\"${LLVM_ROOT_REMOTE}\" LLVM_ROOT=\"${LLVM_ROOT_REMOTE}\" && export LD_LIBRARY_PATH=\"${MPI_LIB_OVERRIDE}:\\\${LD_LIBRARY_PATH}\"; exec \"\${target_bin}\""
}

${role_case}
EOF
  chmod +x "${role_path}"
}

run_baseline() { # mode (posix|mpiio)
  local mode="$1"
  local ifname="posix-pwrite"
  [[ "${mode}" == "mpiio" ]] && ifname="mpiio"
  local log_file="${COMPARE_OUTDIR}/${mode}.log"
  local data_path="${COMPARE_OUTDIR}/data-${mode}"
  local role_path="${COMPARE_OUTDIR}/role-${mode}.sh"
  echo "[amd-hacc-io-cmp] baseline ${mode} (${COMPARE_TASKS} SPMD ranks)"
  # run_in_container execs the target directly — wrap binary+args in a helper:
  cat > "${COMPARE_OUTDIR}/exec-${mode}.sh" <<EOF
#!/usr/bin/env bash
exec "${BASELINE_BIN}" "${COMPARE_PARTICLES}" "${data_path}"
EOF
  chmod +x "${COMPARE_OUTDIR}/exec-${mode}.sh"
  # HACC_BASELINE_FSYNC=1 → storage-fair: baseline fsyncs at checkpoint close
  # (matches the OMPFILE close policy) instead of returning on page-cache fill.
  write_role_script "${role_path}" \
    "export HACC_IO_INTERFACE=${ifname}
export HACC_FSYNC_ON_CLOSE=${HACC_BASELINE_FSYNC:-0}" \
    "run_in_container \"${COMPARE_OUTDIR}/exec-${mode}.sh\""
  srun --kill-on-bad-exit=1 --mpi="${RUN_MPI_TYPE}" --nodes="${RUN_NODES}" --ntasks="${COMPARE_TASKS}" --ntasks-per-node=1 \
    "${role_path}" > "${log_file}" 2>&1 || { echo "Error: baseline ${mode} failed" >&2; tail -n 40 "${log_file}" >&2; exit 1; }
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
  local role_path="${COMPARE_OUTDIR}/role-${mode}.sh"
  echo "[amd-hacc-io-cmp] ${mode} split-role (1 app + $((COMPARE_TASKS - 1)) proxies, logical_ranks=${COMPARE_TASKS}, concurrency=${conc})"
  cat > "${COMPARE_OUTDIR}/exec-app-${mode}.sh" <<EOF
#!/usr/bin/env bash
export HACC_FILE="${data_path}" HACC_RANKS="${COMPARE_TASKS}" HACC_PARTICLES="${COMPARE_PARTICLES}"
export HACC_CONCURRENCY="${conc}"
exec "${OMPFILE_BIN}"
EOF
  chmod +x "${COMPARE_OUTDIR}/exec-app-${mode}.sh"
  write_role_script "${role_path}" \
    "export OMP_TARGET_OFFLOAD=MANDATORY
export OMPTARGET_DISABLE_HOST_PLUGIN=1
export LIBOMPFILE_BACKEND=MPI
export LIBOMPFILE_MPP_OPEN=1
export LIBOMPFILE_MPP_IO=1
export LIBOMPFILE_MPP_PING=1
export LIBOMPFILE_SCHEDULER=${OMPFILE_SCHEDULER}
export LIBOMPFILE_OPT_OPEN_CACHE=1
export LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN=1
export LIBOMPFILE_OPT_STATS=1
export UCX_TLS=\${UCX_TLS:-all}
export UCX_POSIX_USE_PROC_LINK=\${UCX_POSIX_USE_PROC_LINK:-n}
${extra_env}" \
    "if [[ \"\${SLURM_PROCID}\" -eq \$((SLURM_NTASKS - 1)) ]]; then
  run_in_container \"${COMPARE_OUTDIR}/exec-app-${mode}.sh\"
else
  run_in_container \"${PROXY_BIN}\"
fi"
  srun --kill-on-bad-exit=1 --mpi="${RUN_MPI_TYPE}" --nodes="${RUN_NODES}" --ntasks="${COMPARE_TASKS}" --ntasks-per-node=1 \
    "${role_path}" > "${log_file}" 2>&1 || { echo "Error: ${mode} failed" >&2; tail -n 60 "${log_file}" >&2; exit 1; }
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
  local role_path="${COMPARE_OUTDIR}/role-${mode}.sh"
  echo "[amd-hacc-io-cmp] ${mode} target-region issuers (1 app + $((COMPARE_TASKS - 1)) proxies, logical_ranks=${COMPARE_TASKS}, data generated device-side)"
  cat > "${COMPARE_OUTDIR}/exec-app-${mode}.sh" <<EOF
#!/usr/bin/env bash
export HACC_FILE="${data_path}" HACC_RANKS="${COMPARE_TASKS}" HACC_PARTICLES="${COMPARE_PARTICLES}" HACC_FILE_PER_RANK="${fpr}" HACC_IO_ASYNC="${async}" HACC_SEGMENTED="${segmented}" HACC_CONCURRENT_OPEN="${concurrent}"
exec "${TARGET_BIN}"
EOF
  chmod +x "${COMPARE_OUTDIR}/exec-app-${mode}.sh"
  write_role_script "${role_path}" \
    "export OMP_TARGET_OFFLOAD=MANDATORY
export OMPTARGET_DISABLE_HOST_PLUGIN=1
export LIBOMPFILE_BACKEND=MPI
export LIBOMPFILE_MPP_OPEN=1
export LIBOMPFILE_MPP_IO=1
export LIBOMPFILE_MPP_PING=1
export LIBOMPFILE_SCHEDULER=${OMPFILE_SCHEDULER}
export LIBOMPFILE_OPT_OPEN_CACHE=1
export LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN=1
export LIBOMPFILE_OPT_STATS=1
export UCX_TLS=\${UCX_TLS:-all}
export UCX_POSIX_USE_PROC_LINK=\${UCX_POSIX_USE_PROC_LINK:-n}
${extra_env}" \
    "if [[ \"\${SLURM_PROCID}\" -eq \$((SLURM_NTASKS - 1)) ]]; then
  run_in_container \"${COMPARE_OUTDIR}/exec-app-${mode}.sh\"
else
  run_in_container \"${PROXY_BIN}\"
fi"
  srun --kill-on-bad-exit=1 --mpi="${RUN_MPI_TYPE}" --nodes="${RUN_NODES}" --ntasks="${COMPARE_TASKS}" --ntasks-per-node=1 \
    "${role_path}" > "${log_file}" 2>&1 || { echo "Error: ${mode} failed" >&2; tail -n 60 "${log_file}" >&2; exit 1; }
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
  echo "$2,$(get writers),$(get devices),$(get transfer_bytes),$(get block_bytes),$(get total_bytes),$(get write_open_s),$(get write_s),$(get write_close_s),$(get write_mib_s),$(get read_open_s),$(get read_s),$(get read_close_s),$(get read_mib_s),$(get verified)" >> "${IOR_TARGET_CSV}"
}

run_ior_target() { # mode extra_env_stmt
  local mode="$1"
  local extra_env="$2"
  local log_file="${COMPARE_OUTDIR}/${mode}.log"
  local data_path="${COMPARE_OUTDIR}/data-${mode}"
  local role_path="${COMPARE_OUTDIR}/role-${mode}.sh"
  echo "[amd-hacc-io-cmp] ${mode} target-region IOR pattern (writers=proxies, transfer=${IORT_TRANSFER_BYTES} block=${IORT_BLOCK_BYTES})"
  cat > "${COMPARE_OUTDIR}/exec-app-${mode}.sh" <<EOF
#!/usr/bin/env bash
export IORT_FILE="${data_path}" IORT_TRANSFER_BYTES="${IORT_TRANSFER_BYTES}" IORT_BLOCK_BYTES="${IORT_BLOCK_BYTES}"
exec "${IOR_TARGET_BIN}"
EOF
  chmod +x "${COMPARE_OUTDIR}/exec-app-${mode}.sh"
  write_role_script "${role_path}" \
    "export OMP_TARGET_OFFLOAD=MANDATORY
export OMPTARGET_DISABLE_HOST_PLUGIN=1
export LIBOMPFILE_BACKEND=MPI
export LIBOMPFILE_MPP_OPEN=1
export LIBOMPFILE_MPP_IO=1
export LIBOMPFILE_MPP_PING=1
export LIBOMPFILE_SCHEDULER=${OMPFILE_SCHEDULER}
export LIBOMPFILE_OPT_OPEN_CACHE=1
export LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN=1
export LIBOMPFILE_OPT_STATS=1
export UCX_TLS=\${UCX_TLS:-all}
export UCX_POSIX_USE_PROC_LINK=\${UCX_POSIX_USE_PROC_LINK:-n}
${extra_env}" \
    "if [[ \"\${SLURM_PROCID}\" -eq \$((SLURM_NTASKS - 1)) ]]; then
  run_in_container \"${COMPARE_OUTDIR}/exec-app-${mode}.sh\"
else
  run_in_container \"${PROXY_BIN}\"
fi"
  srun --kill-on-bad-exit=1 --mpi="${RUN_MPI_TYPE}" --nodes="${RUN_NODES}" --ntasks="${COMPARE_TASKS}" --ntasks-per-node=1 \
    "${role_path}" > "${log_file}" 2>&1 || { echo "Error: ${mode} failed" >&2; tail -n 60 "${log_file}" >&2; exit 1; }
  grep -q 'CONTENTS VERIFIED' "${log_file}" || { echo "Error: ${mode} did not verify" >&2; tail -n 60 "${log_file}" >&2; exit 1; }
  collect_ior_target_summary "${log_file}" "${mode}"
  if [[ "${COMPARE_CLEAN_DATA}" == "1" ]]; then
    rm -f "${data_path}"*
  fi
}

# Topology for the write-back lane: node-local stage on proxy hosts,
# authoritative storage on wekafs (/work1).
TOPOLOGY_FILE="${COMPARE_OUTDIR}/topology.txt"
{
  echo "global /work1 parallel_fs"
  scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n $((COMPARE_TASKS - 1)) | while read -r host; do
    echo "host ${host} /tmp/${USER}/libompfile-stage/hacc-${SLURM_JOB_ID} local-posix"
  done
} > "${TOPOLOGY_FILE}"

WB_ENV="export LIBOMPFILE_STAGE_MODE=readthrough
export LIBOMPFILE_STAGE_ROOT_POLICY=topology
export LIBOMPFILE_TOPOLOGY_FILE=${TOPOLOGY_FILE}
export LIBOMPFILE_STAGE_WRITE_MODE=write-back
export LIBOMPFILE_STAGE_DIRTY_WATERMARK_BYTES=8388608
export LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY=close"
WT_ENV="export LIBOMPFILE_STAGE_MODE=off
export LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY=each"
# Fair-posture write-through: baselines never fsync per write, so the
# target-region lanes default to the close policy (override via env).
TARGET_WT_ENV="export LIBOMPFILE_STAGE_MODE=off
export LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY=${HACC_TARGET_FSYNC_POLICY:-close}"
# Owner-bypass: shared file, each proxy writes its disjoint range locally.
TARGET_LOCAL_ENV="export LIBOMPFILE_STAGE_MODE=off
export LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY=${HACC_TARGET_FSYNC_POLICY:-close}
export LIBOMPFILE_OPT_LOCAL_DISJOINT_WRITE=1"

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
    ompfile-target-fpr-concurrent) run_ompfile_target ompfile-target-fpr-concurrent "${TARGET_LOCAL_ENV}" 1 0 0 1 ;;
    ior-target) run_ior_target ior-target "${TARGET_WT_ENV}" ;;
    *) echo "Error: unsupported mode '${mode}' in HACC_COMPARE_MODES." >&2; exit 1 ;;
  esac
done

echo "[amd-hacc-io-cmp] summary:"
cat "${SUMMARY_CSV}"
if [[ "$(wc -l < "${IOR_TARGET_CSV}")" -gt 1 ]]; then
  cat "${IOR_TARGET_CSV}"
fi

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

echo "[amd-hacc-io-cmp] status=PASS summary_csv=${SUMMARY_CSV}"
