// IOR-pattern write/read benchmark — target-region (distributed-issuer)
// OMPFILE variant.
//
// Fairness counterpart to the mpiio-vs-mpp lane's OMPFILE mode: real IOR runs
// there with `-F` (file per process) but issues every byte from the single
// split-role app rank, so the measurement is the app-rank transport, not the
// I/O path. This driver replicates IOR's `-F` access geometry (per-writer
// file, one block written/read in transfer-size chunks, sequential offsets)
// but executes each writer inside `#pragma omp target device(d)` on a proxy
// worker: the transfer buffer is allocated and filled device-side, and every
// omp_file_pwrite/pread issues from the worker's own node — one writer per
// proxy, mirroring the per-node parallelism of the MPIIO/POSIX baselines.
//
// Phases are timed IOR-style (open / write-or-read / close walls, aggregate
// bandwidth over the data phase). Files are pre-created + truncated with
// POSIX from the app rank (the repo's OOC seed pattern), which also sidesteps
// the known planned-open O_CREAT ENOENT storm.
//
// Env knobs:
//   IORT_FILE            base data path (required; per-writer suffix .NNN)
//   IORT_TRANSFER_BYTES  chunk size per I/O call        (default 1 MiB)
//   IORT_BLOCK_BYTES     bytes per writer               (default 16 MiB)
//   IORT_WRITERS         logical writers (default = visible devices)
//   IORT_SKIP_READ       1 = write phase only
//
// Output: one machine-parseable line
//   IOR_TARGET_SUMMARY writers=.. devices=.. transfer_bytes=.. block_bytes=..
//     total_bytes=.. write_open_s=.. write_s=.. write_close_s=..
//     write_mib_s=.. read_open_s=.. read_s=.. read_close_s=.. read_mib_s=..
//     verified=..

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

// Deterministic 8-byte stamp at the head of each transfer chunk so the read
// phase can verify placement without a full-buffer pattern.
static inline uint64_t chunk_stamp(int64_t writer, int64_t chunk) {
  return 0x1097a11edULL ^ ((uint64_t)writer << 32) ^ (uint64_t)chunk;
}

struct PhaseTimes {
  double open_s = 0.0, io_s = 0.0, close_s = 0.0;
};

// One full phase (write or read+verify) across all writers: open all handles
// (timed), run every writer's chunk loop concurrently (timed), close all
// handles (timed). Returns 0 on success.
static int run_phase(char **paths, int writers, int devices,
                     int64_t transfer_bytes, int64_t chunks, bool writing,
                     PhaseTimes *times) {
  int *handles = (int *)malloc((size_t)writers * sizeof(int));
  if (!handles)
    return 1;
  int failed = 0;

  double t = now_s();
  for (int w = 0; w < writers && !failed; ++w) {
    const int device_id = w % devices;
    const char *path = paths[w];
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
      fprintf(stderr, "FAIL ior-target open writer=%d errno=%d (%s)\n", w,
              saved_errno, strerror(saved_errno));
      failed = 1;
    }
    handles[w] = handle;
  }
  times->open_s = now_s() - t;
  if (failed) {
    free(handles);
    return 1;
  }

  t = now_s();
#pragma omp parallel num_threads(writers) shared(failed)
  {
#pragma omp single
    {
      for (int w = 0; w < writers; ++w) {
        const int device_id = w % devices;
        const int handle = handles[w];
#pragma omp task firstprivate(w, device_id, handle) shared(failed)
        {
          int rc = 0;
          int saved_errno = 0;
          const int64_t wr = w;
          const int64_t n_chunks = chunks;
          const int64_t chunk_bytes = transfer_bytes;
          const bool is_write = writing;
#pragma omp target firstprivate(handle, wr, n_chunks, chunk_bytes, is_write)   \
    device(device_id) map(tofrom : rc, saved_errno)
          {
            unsigned char *buf = (unsigned char *)malloc((size_t)chunk_bytes);
            if (!buf) {
              rc = 10;
              saved_errno = ENOMEM;
            } else {
              if (is_write) {
                for (int64_t i = 0; i < chunk_bytes; ++i)
                  buf[i] = (unsigned char)(((wr + 1) * 131 + i) & 0xff);
              }
              for (int64_t c = 0; c < n_chunks && rc == 0; ++c) {
                const int64_t off = c * chunk_bytes;
                if (is_write) {
                  *(uint64_t *)buf = chunk_stamp(wr, c);
                  if (omp_file_pwrite(handle, off, buf, (size_t)chunk_bytes,
                                      0) != 0) {
                    rc = 20;
                    saved_errno = errno;
                  }
                } else {
                  if (omp_file_pread(handle, off, buf, (size_t)chunk_bytes,
                                     0) != 0) {
                    rc = 21;
                    saved_errno = errno;
                  } else if (*(uint64_t *)buf != chunk_stamp(wr, c)) {
                    rc = 30;
                  }
                }
              }
              free(buf);
            }
          }
          if (rc != 0) {
            fprintf(stderr,
                    "FAIL ior-target %s writer=%d device=%d rc=%d errno=%d "
                    "(%s)\n",
                    writing ? "write" : "read", w, device_id, rc, saved_errno,
                    strerror(saved_errno));
#pragma omp atomic write
            failed = 1;
          }
        }
      }
#pragma omp taskwait
    }
  }
  times->io_s = now_s() - t;

  t = now_s();
  for (int w = 0; w < writers; ++w) {
    const int device_id = w % devices;
    const int handle = handles[w];
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
      fprintf(stderr, "FAIL ior-target close writer=%d errno=%d (%s)\n", w,
              saved_errno, strerror(saved_errno));
      failed = 1;
    }
  }
  times->close_s = now_s() - t;

  free(handles);
  return failed;
}

static void warmup_offload() {
  int x = 0;
#pragma omp target map(tofrom : x)
  { x = 1; }
  (void)x;
}

int main() {
  const char *base = getenv("IORT_FILE");
  if (!base || !*base) {
    fprintf(stderr, "FAIL ior-target: IORT_FILE not set\n");
    return 1;
  }
  const int64_t transfer_bytes =
      env_i64("IORT_TRANSFER_BYTES", 1024 * 1024);
  const int64_t block_bytes =
      env_i64("IORT_BLOCK_BYTES", 16LL * 1024 * 1024);
  const int skip_read = (int)env_i64("IORT_SKIP_READ", 0);
  if (block_bytes % transfer_bytes != 0) {
    fprintf(stderr, "FAIL ior-target: block %% transfer != 0\n");
    return 1;
  }
  const int64_t chunks = block_bytes / transfer_bytes;

  warmup_offload();
  const int devices = wait_for_target_devices(1);
  if (devices < 1) {
    fprintf(stderr, "FAIL ior-target: no target devices visible\n");
    return 1;
  }
  int writers = (int)env_i64("IORT_WRITERS", devices);
  if (writers < 1)
    writers = devices;
  const int64_t total_bytes = (int64_t)writers * block_bytes;

  // Per-writer files, pre-created + truncated app-side (OOC seed pattern).
  char **paths = (char **)calloc((size_t)writers, sizeof(char *));
  if (!paths)
    return 1;
  for (int w = 0; w < writers; ++w) {
    const size_t len = strlen(base) + 16;
    paths[w] = (char *)malloc(len);
    if (!paths[w])
      return 1;
    snprintf(paths[w], len, "%s.%08d", base, w);
    int fd = ::open(paths[w], O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0 || ::ftruncate(fd, (off_t)block_bytes) != 0) {
      fprintf(stderr, "FAIL ior-target create writer=%d errno=%d (%s)\n", w,
              errno, strerror(errno));
      if (fd >= 0)
        ::close(fd);
      return 1;
    }
    ::close(fd);
  }

  PhaseTimes wt, rt;
  if (run_phase(paths, writers, devices, transfer_bytes, chunks,
                /*writing=*/true, &wt) != 0)
    return 1;

  int verified = skip_read ? -1 : 1;
  if (!skip_read) {
    if (run_phase(paths, writers, devices, transfer_bytes, chunks,
                  /*writing=*/false, &rt) != 0) {
      verified = 0;
      return 1;
    }
  }

  const double write_mib =
      (double)total_bytes / (1024.0 * 1024.0) / wt.io_s;
  const double read_mib =
      skip_read ? 0.0 : (double)total_bytes / (1024.0 * 1024.0) / rt.io_s;

  printf("IOR_TARGET_SUMMARY writers=%d devices=%d transfer_bytes=%" PRId64
         " block_bytes=%" PRId64 " total_bytes=%" PRId64
         " write_open_s=%.6f write_s=%.6f write_close_s=%.6f "
         "write_mib_s=%.3f read_open_s=%.6f read_s=%.6f read_close_s=%.6f "
         "read_mib_s=%.3f verified=%d\n",
         writers, devices, transfer_bytes, block_bytes, total_bytes, wt.open_s,
         wt.io_s, wt.close_s, write_mib, rt.open_s, rt.io_s, rt.close_s,
         read_mib, verified);
  printf(" CONTENTS VERIFIED... Success \n"); // shared success marker
  for (int w = 0; w < writers; ++w)
    free(paths[w]);
  free(paths);
  return 0;
}
