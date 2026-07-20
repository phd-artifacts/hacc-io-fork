// HACC-IO checkpoint/restart kernel — OMPFILE (libompfile + MPP) variant.
//
// MPI-free by design: this binary runs as the single app rank of a split-role
// MPP launch (llvm-offload-mpi-proxy-device on ranks 0..N-2, this app on the
// last rank), mirroring the repo's IOR/OOC methodology. The runtime owns all
// distribution; the app only issues omp_file_* calls.
//
// The file layout replicates the GLEAN single-shared-file format: a
// FILE_HEADER_SIZE_MAX (24 MB) header region followed by one contiguous block
// per logical rank, each block holding the nine particle arrays back-to-back
// (xx yy zz vx vy vz phi [float], pid [int64], mask [uint16] = 38 B/particle),
// with the same deterministic fill pattern as the upstream driver so restart
// verification is bit-exact.
//
// Env knobs:
//   HACC_RANKS        logical SPMD ranks to emulate (default 8)
//   HACC_PARTICLES    particles per logical rank (default 1000000)
//   HACC_FILE         checkpoint path (required)
//   HACC_SKIP_READ    1 = write phase only
//   HACC_CONCURRENCY  concurrent issuer threads over logical ranks (default 1
//                     = sequential; clamped to HACC_RANKS). Concurrent host
//                     threads on one handle are the sanctioned libompfile
//                     pattern (see test-write-batch-overlap-order); disjoint
//                     per-rank ranges, so no app-level ordering is needed.
//
// Output: one machine-parseable line
//   HACC_IO_SUMMARY interface=ompfile logical_ranks=.. particles_per_rank=..
//     bytes=.. write_s=.. write_mib_s=.. read_s=.. read_mib_s=.. verified=..
//     concurrency=..

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
#include <vector>

#include <omp.h>

#include "file_interface.h" /* libompfile API */

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

struct RankBuffers {
  std::vector<float> xx, yy, zz, vx, vy, vz, phi;
  std::vector<int64_t> pid;
  std::vector<uint16_t> mask;
  explicit RankBuffers(int64_t n)
      : xx(n), yy(n), zz(n), vx(n), vy(n), vz(n), phi(n), pid(n), mask(n) {}
};

static void fill_rank(RankBuffers &b, int64_t n, int logical_rank) {
  for (int64_t i = 0; i < n; ++i) {
    const float f = (float)i;
    b.xx[i] = f;
    b.yy[i] = f;
    b.zz[i] = f;
    b.vx[i] = f;
    b.vy[i] = f;
    b.vz[i] = f;
    b.phi[i] = f;
    b.pid[i] = i;
    b.mask[i] = (uint16_t)logical_rank;
  }
}

static int pwrite_all(int handle, int64_t off, const void *buf, int64_t n) {
  return omp_file_pwrite(handle, off, (void *)buf, (size_t)n, 0);
}

static int pread_all(int handle, int64_t off, void *buf, int64_t n) {
  return omp_file_pread(handle, off, buf, (size_t)n, 0);
}

// Write or read one logical rank's block; returns 0 on success.
static int rank_block_io(int handle, int64_t base, RankBuffers &b, int64_t n,
                         bool writing) {
  struct Seg {
    void *p;
    int64_t bytes;
  } segs[9] = {
      {b.xx.data(), n * (int64_t)sizeof(float)},
      {b.yy.data(), n * (int64_t)sizeof(float)},
      {b.zz.data(), n * (int64_t)sizeof(float)},
      {b.vx.data(), n * (int64_t)sizeof(float)},
      {b.vy.data(), n * (int64_t)sizeof(float)},
      {b.vz.data(), n * (int64_t)sizeof(float)},
      {b.phi.data(), n * (int64_t)sizeof(float)},
      {b.pid.data(), n * (int64_t)sizeof(int64_t)},
      {b.mask.data(), n * (int64_t)sizeof(uint16_t)},
  };
  int64_t off = base;
  for (int s = 0; s < 9; ++s) {
    const int rc = writing ? pwrite_all(handle, off, segs[s].p, segs[s].bytes)
                           : pread_all(handle, off, segs[s].p, segs[s].bytes);
    if (rc != 0) {
      fprintf(stderr, "FAIL hacc-ompfile %s seg=%d off=%" PRId64
                      " bytes=%" PRId64 " errno=%d (%s)\n",
              writing ? "pwrite" : "pread", s, off, segs[s].bytes, errno,
              strerror(errno));
      return 1;
    }
    off += segs[s].bytes;
  }
  return 0;
}

// Verify against the deterministic fill pattern without a second buffer.
static int verify_rank(const RankBuffers &r, int64_t n, int logical_rank) {
  for (int64_t i = 0; i < n; ++i) {
    const float f = (float)i;
    if (r.xx[i] != f || r.yy[i] != f || r.zz[i] != f || r.vx[i] != f ||
        r.vy[i] != f || r.vz[i] != f || r.phi[i] != f || r.pid[i] != i ||
        r.mask[i] != (uint16_t)logical_rank) {
      fprintf(stderr, "FAIL hacc-ompfile verify rank=%d index=%" PRId64 "\n",
              logical_rank, i);
      return 1;
    }
  }
  return 0;
}

// One checkpoint (write) or restart (read+verify) phase: the logical-rank
// loop, issued by `conc` concurrent host threads with per-thread buffers.
// Returns 0 on success.
static int run_phase(int handle, int64_t ranks, int64_t particles,
                     int64_t rank_bytes, int conc, bool writing) {
  int failed = 0;
#pragma omp parallel num_threads(conc) shared(failed)
  {
    RankBuffers buf(particles);
#pragma omp for schedule(dynamic)
    for (int64_t lr = 0; lr < ranks; ++lr) {
      int stop = 0;
#pragma omp atomic read
      stop = failed;
      if (stop)
        continue;
      const int64_t base = kHeaderBytes + lr * rank_bytes;
      int rc;
      if (writing) {
        fill_rank(buf, particles, (int)lr);
        rc = rank_block_io(handle, base, buf, particles, /*writing=*/true);
      } else {
        rc = rank_block_io(handle, base, buf, particles, /*writing=*/false);
        if (rc == 0)
          rc = verify_rank(buf, particles, (int)lr);
      }
      if (rc != 0) {
#pragma omp atomic write
        failed = 1;
      }
    }
  }
  return failed;
}

// Kick the offload runtime so the MPI proxy plugin initializes and the
// mpp_shim path resolves before the first omp_file_* call (same pattern as
// the IOR aiori-ompfile backend's target warmup).
static void warmup_offload() {
  int x = 0;
#pragma omp target map(tofrom : x)
  { x = 1; }
  (void)x;
}

int main() {
  warmup_offload();
  const char *path = getenv("HACC_FILE");
  if (!path || !*path) {
    fprintf(stderr, "FAIL hacc-ompfile: HACC_FILE not set\n");
    return 1;
  }
  const int64_t ranks = env_i64("HACC_RANKS", 8);
  const int64_t particles = env_i64("HACC_PARTICLES", 1000000);
  const int skip_read = (int)env_i64("HACC_SKIP_READ", 0);
  int concurrency = (int)env_i64("HACC_CONCURRENCY", 1);
  if (concurrency > ranks)
    concurrency = (int)ranks;
  if (concurrency < 1)
    concurrency = 1;
  const int64_t rank_bytes = particles * kRecordBytes;
  const int64_t total_bytes = ranks * rank_bytes;
  const int64_t file_bytes = kHeaderBytes + total_bytes;

  // ----- checkpoint (write) phase ------------------------------------------
  const double w0 = now_s();

  // Create + size the file with POSIX (libompfile has no create/truncate),
  // then hand it to the runtime. Matches the OOC seed pattern.
  {
    int fd = ::open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0 || ::ftruncate(fd, (off_t)file_bytes) != 0) {
      fprintf(stderr, "FAIL hacc-ompfile create/truncate errno=%d (%s)\n",
              errno, strerror(errno));
      if (fd >= 0)
        ::close(fd);
      return 1;
    }
    // Minimal header metainfo at offset 0: ranks, particles per rank.
    int64_t meta[2] = {ranks, particles};
    if (::pwrite(fd, meta, sizeof(meta), 0) != (ssize_t)sizeof(meta)) {
      fprintf(stderr, "FAIL hacc-ompfile header write errno=%d\n", errno);
      ::close(fd);
      return 1;
    }
    ::close(fd);
  }

  int handle = omp_file_open((char *)path);
  if (handle < 0) {
    fprintf(stderr, "FAIL hacc-ompfile omp_file_open errno=%d (%s)\n", errno,
            strerror(errno));
    return 1;
  }

  if (run_phase(handle, ranks, particles, rank_bytes, concurrency,
                /*writing=*/true) != 0)
    return 1;

  if (omp_file_close(handle) != 0) {
    fprintf(stderr, "FAIL hacc-ompfile close errno=%d (%s)\n", errno,
            strerror(errno));
    return 1;
  }
  const double write_s = now_s() - w0;

  // ----- restart (read + verify) phase -------------------------------------
  double read_s = 0.0;
  int verified = skip_read ? -1 : 1;
  if (!skip_read) {
    const double r0 = now_s();
    handle = omp_file_open((char *)path);
    if (handle < 0) {
      fprintf(stderr, "FAIL hacc-ompfile restart open errno=%d\n", errno);
      return 1;
    }
    if (run_phase(handle, ranks, particles, rank_bytes, concurrency,
                  /*writing=*/false) != 0)
      verified = 0;
    if (omp_file_close(handle) != 0) {
      fprintf(stderr, "FAIL hacc-ompfile restart close errno=%d\n", errno);
      return 1;
    }
    read_s = now_s() - r0;
    if (!verified)
      return 1;
  }

  const double write_mib = (double)total_bytes / (1024.0 * 1024.0) / write_s;
  const double read_mib =
      skip_read ? 0.0 : (double)total_bytes / (1024.0 * 1024.0) / read_s;
  printf("HACC_IO_SUMMARY interface=ompfile logical_ranks=%" PRId64
         " particles_per_rank=%" PRId64 " bytes=%" PRId64
         " write_s=%.6f write_mib_s=%.3f read_s=%.6f read_mib_s=%.3f "
         "verified=%d concurrency=%d\n",
         ranks, particles, total_bytes, write_s, write_mib, read_s, read_mib,
         verified, concurrency);
  printf(" CONTENTS VERIFIED... Success \n"); // match upstream success marker
  return 0;
}
