// HACC-IO checkpoint/restart kernel — target-region (distributed-issuer)
// OMPFILE variant.
//
// Fairness counterpart to testhacc_ompfile.cc: instead of the single app rank
// funnelling every byte through the MPP transport, each logical rank's block
// I/O executes inside `#pragma omp target device(d)` on a proxy worker, the
// way tasks run in the deployment model. The particle arrays are generated
// *on the device* (the worker that would have computed them), so the
// checkpoint payload never crosses the app-rank transport — each proxy fills
// its buffer locally and issues omp_file_pwrite from its own node, mirroring
// the per-node parallelism of the SPMD POSIX/MPI-IO baselines.
//
// File layout, fill pattern, and verification are bit-identical to
// testhacc_ompfile.cc (GLEAN single shared file: 24 MB header + one
// contiguous 38 B/particle block per logical rank).
//
// Scheduling: logical ranks are issued in waves of `devices` so at most one
// region (one rank-sized buffer) is in flight per device; within a wave all
// devices run concurrently via host tasks (the sanctioned pattern from
// test-target-region-staging / test-ooc-cholesky-io).
//
// Env knobs:
//   HACC_FILE           checkpoint path (required)
//   HACC_RANKS          logical SPMD ranks to emulate (default 8)
//   HACC_PARTICLES      particles per logical rank (default 1000000)
//   HACC_SKIP_READ      1 = write phase only
//   HACC_FILE_PER_RANK  1 = one file per logical rank (${HACC_FILE}.rNNN, no
//                       header, segments from offset 0), opened/closed inside
//                       the issuing device's region. Diagnostic for shared-file
//                       owner forwarding: per-rank files make the opening proxy
//                       the owner, so every pwrite executes node-locally.
//
// Output: one machine-parseable line
//   HACC_IO_SUMMARY interface=ompfile_target logical_ranks=..
//     particles_per_rank=.. bytes=.. write_s=.. write_mib_s=.. read_s=..
//     read_mib_s=.. verified=.. concurrency=<devices>

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <omp.h>

#pragma omp declare target
#include "file_interface.h" /* libompfile API, resolved proxy-side */
#pragma omp end declare target

static const int64_t kHeaderBytes = 25165824; // FILE_HEADER_SIZE_MAX (24 MB)
static const int64_t kRecordBytes =
    7 * (int64_t)sizeof(float) + sizeof(int64_t) + sizeof(uint16_t); // 38

static double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int64_t env_i64(const char *name, int64_t defval) {
  const char *v = getenv(name);
  if (!v || !*v)
    return defval;
  char *end = nullptr;
  long long parsed = strtoll(v, &end, 10);
  return (end && *end == '\0' && parsed > 0) ? (int64_t)parsed : defval;
}

static int wait_for_target_devices(int required) {
  const int retries = (int)env_i64("OMPFILE_DEVICE_DISCOVERY_RETRIES", 50);
  const int sleep_ms = (int)env_i64("OMPFILE_DEVICE_DISCOVERY_SLEEP_MS", 100);
  int devices = 0;
  for (int attempt = 0; attempt <= retries; ++attempt) {
    devices = omp_get_num_devices();
    if (devices >= required)
      return devices;
    if (attempt != retries && sleep_ms > 0) {
      struct timespec delay = {sleep_ms / 1000,
                               (long)(sleep_ms % 1000) * 1000000L};
      nanosleep(&delay, nullptr);
    }
  }
  return devices;
}

// One logical rank's block, executed entirely on `device_id`: allocate the
// rank buffer proxy-side, fill (write) or pread+verify (read), and issue the
// nine segment I/Os at the GLEAN offsets. If `fpr_path` is non-null the region
// opens/closes that per-rank file itself (file-per-rank diagnostic mode) and
// `handle` is ignored. Returns 0 on success; error codes mirror
// testhacc_ompfile (1x=alloc, 2x=io, 3x=verify, 4x=fpr open/close).
static int rank_block_target(int device_id, int handle, int64_t lr,
                             int64_t particles, int64_t base, bool writing,
                             int io_async, int segmented, int *out_errno) {
  int rc = 0;
  int saved_errno = 0;
  const int64_t n = particles;
#pragma omp target firstprivate(handle, lr, n, base, writing, io_async,        \
                                segmented)                                     \
    device(device_id) map(tofrom : rc, saved_errno)
  {
    int active_handle = handle;
    const int64_t float_bytes = n * (int64_t)sizeof(float);
    const int64_t pid_bytes = n * (int64_t)sizeof(int64_t);
    const int64_t mask_bytes = n * (int64_t)sizeof(uint16_t);
    const int64_t rank_bytes = 7 * float_bytes + pid_bytes + mask_bytes;
    unsigned char *buf = (unsigned char *)malloc((size_t)rank_bytes);
    if (!buf) {
      rc = 10;
      saved_errno = ENOMEM;
    } else {
      float *farr[7];
      for (int a = 0; a < 7; ++a)
        farr[a] = (float *)(buf + (int64_t)a * float_bytes);
      int64_t *pid = (int64_t *)(buf + 7 * float_bytes);
      uint16_t *mask = (uint16_t *)(buf + 7 * float_bytes + pid_bytes);

      if (writing) {
        // Data is born on the worker: same deterministic pattern as the
        // upstream driver, generated device-side.
        for (int64_t i = 0; i < n; ++i) {
          const float f = (float)i;
          for (int a = 0; a < 7; ++a)
            farr[a][i] = f;
          pid[i] = i;
          mask[i] = (uint16_t)lr;
        }
      }

      // The nine GLEAN segments (xx..phi, pid, mask) are contiguous in both the
      // buffer and the file, so coalesce them into ONE pwrite/pread of the whole
      // rank block instead of 9 sequential per-op transfers — lifts per-proxy
      // efficiency (each op carries scheduler/transport fixed cost). async lets
      // the runtime pipeline the transfer. segmented=1 falls back to the 9-op
      // GLEAN path for comparison; segmented=N (N>1) splits the rank block
      // into N equal chunks instead, the segment-shape sweep dimension.
      // (io_async/segmented read host-side, passed in.)
      if (!segmented) {
        const int io_rc =
            writing ? omp_file_pwrite(active_handle, base, buf,
                                      (size_t)rank_bytes, io_async)
                    : omp_file_pread(active_handle, base, buf,
                                     (size_t)rank_bytes, io_async);
        if (io_rc != 0) {
          rc = writing ? 20 : 21;
          saved_errno = errno;
        }
      } else if (segmented > 1) {
        const int64_t seg_n = segmented;
        const int64_t chunk = (rank_bytes + seg_n - 1) / seg_n;
        int64_t off = base;
        int64_t buf_off = 0;
        while (buf_off < rank_bytes && rc == 0) {
          const int64_t this_bytes =
              (rank_bytes - buf_off < chunk) ? (rank_bytes - buf_off) : chunk;
          const int io_rc =
              writing ? omp_file_pwrite(active_handle, off, buf + buf_off,
                                        (size_t)this_bytes, io_async)
                      : omp_file_pread(active_handle, off, buf + buf_off,
                                       (size_t)this_bytes, io_async);
          if (io_rc != 0) {
            rc = writing ? 20 : 21;
            saved_errno = errno;
          }
          off += this_bytes;
          buf_off += this_bytes;
        }
      } else {
        const int64_t seg_bytes[9] = {float_bytes, float_bytes, float_bytes,
                                      float_bytes, float_bytes, float_bytes,
                                      float_bytes, pid_bytes,   mask_bytes};
        int64_t off = base;
        int64_t buf_off = 0;
        for (int s = 0; s < 9 && rc == 0; ++s) {
          const int io_rc =
              writing ? omp_file_pwrite(active_handle, off, buf + buf_off,
                                        (size_t)seg_bytes[s], io_async)
                      : omp_file_pread(active_handle, off, buf + buf_off,
                                       (size_t)seg_bytes[s], io_async);
          if (io_rc != 0) {
            rc = writing ? 20 : 21;
            saved_errno = errno;
          }
          off += seg_bytes[s];
          buf_off += seg_bytes[s];
        }
      }

      if (!writing && rc == 0) {
        for (int64_t i = 0; i < n && rc == 0; ++i) {
          const float f = (float)i;
          for (int a = 0; a < 7; ++a) {
            if (farr[a][i] != f) {
              rc = 30;
              break;
            }
          }
          if (rc == 0 && (pid[i] != i || mask[i] != (uint16_t)lr))
            rc = 30;
        }
      }
      free(buf);
    }
  }
  if (out_errno)
    *out_errno = saved_errno;
  return rc;
}

// Open (or close) one handle per device inside a target region on that
// device; the per-device handle is shared by that device's waves (disjoint
// per-rank ranges — the sanctioned concurrent pattern).
static int open_device_handles(const char *path, int devices, int *handles) {
  const size_t path_len = strlen(path) + 1;
  for (int d = 0; d < devices; ++d) {
    int handle = -1;
    int saved_errno = 0;
#pragma omp target device(d) map(to : path[0:path_len])                        \
    map(tofrom : handle, saved_errno)
    {
      handle = omp_file_open(path);
      if (handle < 0)
        saved_errno = errno;
    }
    if (handle < 0) {
      fprintf(stderr, "FAIL hacc-target open device=%d errno=%d (%s)\n", d,
              saved_errno, strerror(saved_errno));
      return 1;
    }
    handles[d] = handle;
  }
  return 0;
}

static int close_device_handles(int devices, const int *handles) {
  for (int d = 0; d < devices; ++d) {
    int rc = 0;
    int saved_errno = 0;
    const int handle = handles[d];
#pragma omp target firstprivate(handle) device(d)                              \
    map(tofrom : rc, saved_errno)
    {
      if (handle >= 0 && omp_file_close(handle) != 0) {
        rc = 1;
        saved_errno = errno;
      }
    }
    if (rc != 0) {
      fprintf(stderr, "FAIL hacc-target close device=%d errno=%d (%s)\n", d,
              saved_errno, strerror(saved_errno));
      return 1;
    }
  }
  return 0;
}

// Open one handle per logical rank's file (file-per-rank mode), sequentially,
// each on its issuing device. Opening up front — outside the concurrent write
// wave — avoids the concurrent-fresh-open path in the runtime (a headnode
// sched request per open) that deadlocked when many devices opened distinct
// fresh files at once. This mirrors the shared-file open_device_handles
// pattern, which never hung.
static int open_rank_handles(char **paths, int64_t ranks, int devices,
                             int *handles) {
  for (int64_t lr = 0; lr < ranks; ++lr) {
    const int device_id = (int)(lr % devices);
    const char *path = paths[lr];
    const size_t path_len = strlen(path) + 1;
    int handle = -1;
    int saved_errno = 0;
#pragma omp target device(device_id) map(to : path[0:path_len])                \
    map(tofrom : handle, saved_errno)
    {
      handle = omp_file_open(path);
      if (handle < 0)
        saved_errno = errno;
    }
    if (handle < 0) {
      fprintf(stderr, "FAIL hacc-target fpr open lr=%" PRId64
                      " device=%d errno=%d (%s)\n",
              lr, device_id, saved_errno, strerror(saved_errno));
      return 1;
    }
    handles[lr] = handle;
  }
  return 0;
}

static int close_rank_handles(int64_t ranks, int devices, const int *handles) {
  for (int64_t lr = 0; lr < ranks; ++lr) {
    const int device_id = (int)(lr % devices);
    const int handle = handles[lr];
    int rc = 0;
    int saved_errno = 0;
#pragma omp target firstprivate(handle) device(device_id)                      \
    map(tofrom : rc, saved_errno)
    {
      if (handle >= 0 && omp_file_close(handle) != 0) {
        rc = 1;
        saved_errno = errno;
      }
    }
    if (rc != 0) {
      fprintf(stderr, "FAIL hacc-target fpr close lr=%" PRId64 " errno=%d (%s)\n",
              lr, saved_errno, strerror(saved_errno));
      return 1;
    }
  }
  return 0;
}

// Concurrent fresh-open path (opt-in via HACC_CONCURRENT_OPEN=1): launch a wave
// of `devices` target-region opens in flight at once — the file-per-rank pattern
// that used to deadlock the runtime by monopolizing each rank's single
// data-event handler on the ENOENT transient-retry loop. Used to validate the
// proxy-side O_CREAT create-on-write fix; the default remains the sequential
// pre-open in open_rank_handles.
static int open_rank_handles_concurrent(char **paths, int64_t ranks, int devices,
                                        int *handles) {
  int failed = 0;
#pragma omp parallel num_threads(devices) shared(failed, handles)
  {
#pragma omp single
    {
      for (int64_t lr = 0; lr < ranks; ++lr) {
        const int device_id = (int)(lr % devices);
        char *path = paths[lr];
#pragma omp task firstprivate(lr, device_id, path) shared(failed, handles)
        {
          int stop = 0;
#pragma omp atomic read
          stop = failed;
          if (!stop) {
            const size_t path_len = strlen(path) + 1;
            int handle = -1;
            int saved_errno = 0;
#pragma omp target device(device_id) map(to : path[0:path_len])                \
    map(tofrom : handle, saved_errno)
            {
              handle = omp_file_open(path);
              if (handle < 0)
                saved_errno = errno;
            }
            if (handle < 0) {
              fprintf(stderr, "FAIL hacc-target fpr concurrent-open lr=%" PRId64
                              " device=%d errno=%d (%s)\n",
                      lr, device_id, saved_errno, strerror(saved_errno));
#pragma omp atomic write
              failed = 1;
            } else {
              handles[lr] = handle;
            }
          }
        }
        // At most `devices` opens in flight per wave (matches run_phase).
        if ((lr + 1) % devices == 0) {
#pragma omp taskwait
        }
      }
#pragma omp taskwait
    }
  }
  return failed;
}

// One checkpoint (write) or restart (read+verify) phase across all logical
// ranks, waves of `devices` regions in flight. `per_rank` selects file-per-rank
// mode: `handles` is indexed by logical rank (offsets from 0), otherwise by
// device (shared file, GLEAN offsets). All handles are pre-opened; the wave
// only issues reads/writes.
static int run_phase(const int *handles, int devices, int64_t ranks,
                     int64_t particles, int64_t rank_bytes, bool writing,
                     bool per_rank, int io_async, int segmented) {
  int failed = 0;
#pragma omp parallel num_threads(devices) shared(failed)
  {
#pragma omp single
    {
      for (int64_t lr = 0; lr < ranks; ++lr) {
        const int device_id = (int)(lr % devices);
#pragma omp task firstprivate(lr, device_id) shared(failed)
        {
          int stop = 0;
#pragma omp atomic read
          stop = failed;
          if (!stop) {
            int io_errno = 0;
            const int64_t base =
                per_rank ? 0 : kHeaderBytes + lr * rank_bytes;
            const int handle = per_rank ? handles[lr] : handles[device_id];
            const int rc = rank_block_target(
                device_id, handle, lr, particles, base, writing, io_async,
                segmented, &io_errno);
            if (rc != 0) {
              fprintf(stderr,
                      "FAIL hacc-target %s lr=%" PRId64
                      " device=%d rc=%d errno=%d (%s)\n",
                      writing ? "write" : "read", lr, device_id, rc, io_errno,
                      strerror(io_errno));
#pragma omp atomic write
              failed = 1;
            }
          }
        }
        // At most one region in flight per device: wait between waves.
        if ((lr + 1) % devices == 0) {
#pragma omp taskwait
        }
      }
#pragma omp taskwait
    }
  }
  return failed;
}

static void warmup_offload() {
  int x = 0;
#pragma omp target map(tofrom : x)
  { x = 1; }
  (void)x;
}

int main() {
  const char *path = getenv("HACC_FILE");
  if (!path || !*path) {
    fprintf(stderr, "FAIL hacc-target: HACC_FILE not set\n");
    return 1;
  }
  const int64_t ranks = env_i64("HACC_RANKS", 8);
  const int64_t particles = env_i64("HACC_PARTICLES", 1000000);
  const int skip_read = (int)env_i64("HACC_SKIP_READ", 0);
  const int file_per_rank = (int)env_i64("HACC_FILE_PER_RANK", 0);
  // #3 per-proxy efficiency knobs (read host-side, passed into target regions):
  // coalesce the 9 GLEAN segments into one op (default). Async issue pipelines
  // the write wave (runtime AsyncWriteEngine); HACC_IO_ASYNC=1 is opt-in.
  const int io_async = (int)env_i64("HACC_IO_ASYNC", 0);
  const int segmented = (int)env_i64("HACC_SEGMENTED", 0);
  // Concurrent fresh-open validation knob: HACC_CONCURRENT_OPEN=1 opens the
  // file-per-rank handles as a concurrent wave (the pattern that used to
  // deadlock) instead of the sequential pre-open workaround. Default 0.
  // Default flipped to 1 (Jul 30, 2026): the concurrent fresh-open deadlock
  // was root-fixed (ENOENT backoff off the single data-event handler) and the
  // path is guarded by the standing hacc-io-concurrent-open lane; concurrent
  // opens are the realistic checkpoint pattern. Set 0 to serialize opens.
  const int concurrent_open = (int)env_i64("HACC_CONCURRENT_OPEN", 1);
  const int64_t rank_bytes = particles * kRecordBytes;
  const int64_t total_bytes = ranks * rank_bytes;
  const int64_t file_bytes = kHeaderBytes + total_bytes;

  warmup_offload();
  int devices = wait_for_target_devices(1);
  if (devices < 1) {
    fprintf(stderr, "FAIL hacc-target: no target devices visible\n");
    return 1;
  }
  if (devices > ranks)
    devices = (int)ranks;

  // Shared file: one handle per device. File-per-rank: one handle per rank.
  // Sequential pre-open is the default; the concurrent-fresh-open runtime
  // deadlock it once dodged is fixed (proxy create-on-write), so
  // HACC_CONCURRENT_OPEN=1 opens the fpr handles in the concurrent wave.
  const int64_t handle_count = file_per_rank ? ranks : devices;
  int *handles = (int *)malloc((size_t)handle_count * sizeof(int));
  char **fpr_paths = nullptr;
  if (!handles)
    return 1;
  if (file_per_rank) {
    fpr_paths = (char **)calloc((size_t)ranks, sizeof(char *));
    if (!fpr_paths)
      return 1;
    for (int64_t lr = 0; lr < ranks; ++lr) {
      const size_t len = strlen(path) + 16;
      fpr_paths[lr] = (char *)malloc(len);
      if (!fpr_paths[lr])
        return 1;
      snprintf(fpr_paths[lr], len, "%s.r%03" PRId64, path, lr);
    }
  }

  // ----- checkpoint (write) phase ------------------------------------------
  const double w0 = now_s();
  if (file_per_rank && !concurrent_open) {
    for (int64_t lr = 0; lr < ranks; ++lr) {
      int fd = ::open(fpr_paths[lr], O_CREAT | O_TRUNC | O_WRONLY, 0644);
      if (fd < 0 || ::ftruncate(fd, (off_t)rank_bytes) != 0) {
        fprintf(stderr, "FAIL hacc-target fpr create lr=%" PRId64
                        " errno=%d (%s)\n",
                lr, errno, strerror(errno));
        if (fd >= 0)
          ::close(fd);
        return 1;
      }
      ::close(fd);
    }
  } else if (file_per_rank) {
    // concurrent_open: leave the files genuinely fresh so the proxy open is the
    // creator — deterministically exercises the create-on-write path that used
    // to ENOENT-storm and deadlock. The write-phase pwrite extends each file.
  } else {
    int fd = ::open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0 || ::ftruncate(fd, (off_t)file_bytes) != 0) {
      fprintf(stderr, "FAIL hacc-target create/truncate errno=%d (%s)\n", errno,
              strerror(errno));
      if (fd >= 0)
        ::close(fd);
      return 1;
    }
    int64_t meta[2] = {ranks, particles};
    if (::pwrite(fd, meta, sizeof(meta), 0) != (ssize_t)sizeof(meta)) {
      fprintf(stderr, "FAIL hacc-target header write errno=%d\n", errno);
      ::close(fd);
      return 1;
    }
    ::close(fd);
  }

  if (file_per_rank
          ? (concurrent_open
                 ? open_rank_handles_concurrent(fpr_paths, ranks, devices,
                                                handles)
                 : open_rank_handles(fpr_paths, ranks, devices, handles))
          : open_device_handles(path, devices, handles))
    return 1;
  if (run_phase(handles, devices, ranks, particles, rank_bytes,
                /*writing=*/true, /*per_rank=*/file_per_rank != 0, io_async,
                segmented) != 0)
    return 1;
  if (file_per_rank ? close_rank_handles(ranks, devices, handles)
                    : close_device_handles(devices, handles))
    return 1;
  const double write_s = now_s() - w0;

  // ----- restart (read + verify) phase -------------------------------------
  double read_s = 0.0;
  int verified = skip_read ? -1 : 1;
  if (!skip_read) {
    const double r0 = now_s();
    if (file_per_rank ? open_rank_handles(fpr_paths, ranks, devices, handles)
                      : open_device_handles(path, devices, handles))
      return 1;
    if (run_phase(handles, devices, ranks, particles, rank_bytes,
                  /*writing=*/false, /*per_rank=*/file_per_rank != 0, io_async,
                  segmented) != 0)
      verified = 0;
    if (file_per_rank ? close_rank_handles(ranks, devices, handles)
                      : close_device_handles(devices, handles))
      return 1;
    read_s = now_s() - r0;
    if (!verified)
      return 1;
  }

  const double write_mib = (double)total_bytes / (1024.0 * 1024.0) / write_s;
  const double read_mib =
      skip_read ? 0.0 : (double)total_bytes / (1024.0 * 1024.0) / read_s;
  printf("HACC_IO_SUMMARY interface=%s logical_ranks=%" PRId64
         " particles_per_rank=%" PRId64 " bytes=%" PRId64
         " write_s=%.6f write_mib_s=%.3f read_s=%.6f read_mib_s=%.3f "
         "verified=%d concurrency=%d\n",
         file_per_rank ? "ompfile_target_fpr" : "ompfile_target", ranks,
         particles, total_bytes, write_s, write_mib, read_s, read_mib, verified,
         devices);
  printf(" CONTENTS VERIFIED... Success \n"); // match upstream success marker
  if (fpr_paths) {
    for (int64_t lr = 0; lr < ranks; ++lr)
      free(fpr_paths[lr]);
    free(fpr_paths);
  }
  free(handles);
  return 0;
}
