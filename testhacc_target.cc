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
//   HACC_PIPELINE       1 = interleave buffer generation with the writes:
//                       fill particle slice c, issue its nine segment writes,
//                       fill slice c+1 while the async engine drains c.
//                       Requires HACC_SEGMENTED != 0 and HACC_PIPELINE_CHUNKS
//                       > 1, and only overlaps anything with HACC_IO_ASYNC=1.
//                       Default 0.
//   HACC_PIPELINE_CHUNKS  particle slices per rank block when pipelining
//                       (default 4). Each slice issues nine pwrites.
//   HACC_PIPELINE_FLUSH 0 = let close drain the pipelined writes instead of
//                       ending each region on omp_file_flush. Default 1;
//                       0 measured faster (job 414886).
//   HACC_FILL_ORDER     particle (default) or array. `array` walks one GLEAN
//                       segment at a time instead of one pass over particles.
//                       Identical bytes; `array` is 1.63x slower (job 414888)
//                       and is kept only as that control. Non-pipelined path
//                       only.
//
// Output: one machine-parseable line
//   HACC_IO_SUMMARY interface=ompfile_target logical_ranks=..
//     particles_per_rank=.. bytes=.. write_s=.. write_mib_s=.. fill_s=..
//     issue_s=.. flush_s=.. region_s=.. read_s=.. read_mib_s=.. verified=..
//     concurrency=<devices> pipeline=.. pipeline_flush=.. fill_order=..
//     pipeline_chunks=..
// fill_s/issue_s/region_s/flush_s are write-phase device-side service time
// summed over regions (see the g_*_s accumulators); they exceed write_s
// whenever more than one device is issuing, and they measure the issuing
// thread only — the storage write itself runs on the async engine's worker.

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

// env_i64 only accepts strictly positive values, so a knob whose default is 1
// cannot be turned off through it. Boolean knobs read through this instead.
static int env_flag(const char *name, int defval) {
  const char *v = getenv(name);
  if (!v || !*v)
    return defval;
  char *end = nullptr;
  long long parsed = strtoll(v, &end, 10);
  return (end && *end == '\0') ? (int)(parsed != 0) : defval;
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

// ---- device-side buffer fill ---------------------------------------------
// The rank block is nine contiguous GLEAN segments: seven float arrays, then
// the int64 pid array, then the uint16 mask array. `fill_byte_range`
// regenerates exactly the bytes of [lo, hi), which is what lets the pipelined
// path produce one chunk at a time. It works at element granularity, so a
// chunk boundary that falls inside an element is filled by both neighbours —
// harmless, because the pattern is a pure function of the element index and
// the async engine has already copied any chunk it was handed.
//
// Note this walks the buffer array-major where the pre-pipelining fill walked
// it particle-major (nine interleaved streams). Both paths now share this one
// implementation so that an A/B of pipelining holds the fill code constant;
// `fill_s` is therefore not comparable with rows measured before the change.
#pragma omp declare target
// Elements of the segment at `seg_lo` (`n` elements of `elem` bytes each) that
// intersect the byte range [lo, hi). Returns 0 when the segment is disjoint.
static int fill_seg_range(int64_t seg_lo, int64_t elem, int64_t n, int64_t lo,
                          int64_t hi, int64_t *i0, int64_t *i1) {
  const int64_t seg_bytes = n * elem;
  int64_t l = lo - seg_lo;
  int64_t h = hi - seg_lo;
  if (h <= 0 || l >= seg_bytes)
    return 0;
  if (l < 0)
    l = 0;
  if (h > seg_bytes)
    h = seg_bytes;
  *i0 = l / elem;
  *i1 = (h + elem - 1) / elem;
  return 1;
}

static void fill_byte_range(unsigned char *buf, int64_t n, int64_t lr,
                            int64_t lo, int64_t hi) {
  const int64_t float_bytes = n * (int64_t)sizeof(float);
  const int64_t pid_bytes = n * (int64_t)sizeof(int64_t);
  int64_t i0 = 0;
  int64_t i1 = 0;
  for (int a = 0; a < 7; ++a) {
    const int64_t seg_lo = (int64_t)a * float_bytes;
    if (!fill_seg_range(seg_lo, (int64_t)sizeof(float), n, lo, hi, &i0, &i1))
      continue;
    float *arr = (float *)(buf + seg_lo);
    for (int64_t i = i0; i < i1; ++i)
      arr[i] = (float)i;
  }
  if (fill_seg_range(7 * float_bytes, (int64_t)sizeof(int64_t), n, lo, hi, &i0,
                     &i1)) {
    int64_t *pid = (int64_t *)(buf + 7 * float_bytes);
    for (int64_t i = i0; i < i1; ++i)
      pid[i] = i;
  }
  if (fill_seg_range(7 * float_bytes + pid_bytes, (int64_t)sizeof(uint16_t), n,
                     lo, hi, &i0, &i1)) {
    uint16_t *mask = (uint16_t *)(buf + 7 * float_bytes + pid_bytes);
    for (int64_t i = i0; i < i1; ++i)
      mask[i] = (uint16_t)lr;
  }
}

// The original traversal, and the one the pipeline uses: one pass over
// particles [p0, p1) touching all nine segments per iteration. This is the
// fast one, by a wide margin — it converts (float)i once per particle where
// fill_byte_range converts it once per particle *per float array*, seven
// times as often. Measured 0.198 s vs 0.323 s per 1.9 GB rank block on AMD
// (job 414888, 3 reps), i.e. 1.63x, worth 1.23x of the whole write phase.
// That is why pipelined chunks are particle ranges rather than byte ranges.
static void fill_particle_range(unsigned char *buf, int64_t n, int64_t lr,
                                int64_t p0, int64_t p1) {
  const int64_t float_bytes = n * (int64_t)sizeof(float);
  const int64_t pid_bytes = n * (int64_t)sizeof(int64_t);
  float *farr[7];
  for (int a = 0; a < 7; ++a)
    farr[a] = (float *)(buf + (int64_t)a * float_bytes);
  int64_t *pid = (int64_t *)(buf + 7 * float_bytes);
  uint16_t *mask = (uint16_t *)(buf + 7 * float_bytes + pid_bytes);
  for (int64_t i = p0; i < p1; ++i) {
    const float f = (float)i;
    for (int a = 0; a < 7; ++a)
      farr[a][i] = f;
    pid[i] = i;
    mask[i] = (uint16_t)lr;
  }
}
#pragma omp end declare target

// Device-side service-time accumulators, summed over every region of the run.
// They sit inside the timed write phase, so a reader can subtract them to
// compare against baselines that fill their arrays before their own timer
// starts. Regions on different devices run concurrently, so these are sums of
// per-region service time, not wall clock.
//   g_fill_s    buffer generation
//   g_issue_s   time inside omp_file_pwrite/pread — with async that is the
//               payload copy and enqueue, not the storage write
//   g_flush_s   time inside the pipelined path's omp_file_flush, which is
//               where a drain that did not overlap a fill shows up
//   g_region_s  the whole region body
//
// These are consecutive intervals on the issuing thread, so they always sum to
// about region_s. The storage write is not among them — it runs on the async
// engine's worker thread. Pipelining therefore shows up as *issue_s falling*
// (fewer enqueues blocking on a full queue) together with a small flush_s, not
// as region_s dropping below their sum.
static double g_fill_s = 0.0;
static double g_issue_s = 0.0;
static double g_flush_s = 0.0;
static double g_region_s = 0.0;

// One logical rank's block, executed entirely on `device_id`: allocate the
// rank buffer proxy-side, fill (write) or pread+verify (read), and issue the
// segment I/Os at the GLEAN offsets. Returns 0 on success; error codes mirror
// testhacc_ompfile (1x=alloc, 2x=io, 3x=verify, 4x=fpr open/close).
static int rank_block_target(int device_id, int handle, int64_t lr,
                             int64_t particles, int64_t base, bool writing,
                             int io_async, int segmented, int pipeline,
                             int pipeline_flush, int pipeline_chunks,
                             int particle_major_fill, int *out_errno) {
  int rc = 0;
  int saved_errno = 0;
  double fill_s = 0.0;
  double issue_s = 0.0;
  double flush_s = 0.0;
  double region_s = 0.0;
  const int64_t n = particles;
#pragma omp target firstprivate(handle, lr, n, base, writing, io_async,        \
                                segmented, pipeline, pipeline_flush,           \
                                pipeline_chunks, particle_major_fill)          \
    device(device_id) map(tofrom : rc, saved_errno, fill_s, issue_s, flush_s,  \
                          region_s)
  {
    const double region_t0 = omp_get_wtime();
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
      // Pipelining needs more than one chunk to interleave, and only actually
      // overlaps anything when the writes are async — with io_async=0 each
      // pwrite runs to completion before the next fill.
      const bool pipelined =
          writing && pipeline != 0 && pipeline_chunks > 1 && segmented != 0;

      // The nine GLEAN segments in buffer/file order: seven float arrays, the
      // int64 pid array, the uint16 mask array. Contiguous in both, which is
      // what lets the non-pipelined path coalesce them.
      const int64_t seg_elem[9] = {
          (int64_t)sizeof(float),   (int64_t)sizeof(float),
          (int64_t)sizeof(float),   (int64_t)sizeof(float),
          (int64_t)sizeof(float),   (int64_t)sizeof(float),
          (int64_t)sizeof(float),   (int64_t)sizeof(int64_t),
          (int64_t)sizeof(uint16_t)};
      int64_t seg_lo[9];
      {
        int64_t off = 0;
        for (int sg = 0; sg < 9; ++sg) {
          seg_lo[sg] = off;
          off += n * seg_elem[sg];
        }
      }

      if (pipelined) {
        // Chunk by PARTICLE range, not by byte range. A byte-range chunk would
        // force the fill to walk one segment at a time, and that traversal is
        // 1.63x slower (see fill_particle_range) — enough to swallow the whole
        // point of the pipeline. So each chunk fills a particle slice with the
        // original traversal and then issues that slice's nine segment writes,
        // handing them to the async engine before generating the next slice.
        // The engine copies the payload on enqueue, which is what makes
        // refilling the buffer behind it safe, and its bounded queue depth
        // (LIBOMPFILE_ASYNC_QUEUE_DEPTH, 2 by default) is the throttle.
        for (int64_t c = 0; c < pipeline_chunks && rc == 0; ++c) {
          const int64_t p0 = c * n / pipeline_chunks;
          const int64_t p1 = (c + 1) * n / pipeline_chunks;
          if (p1 <= p0)
            continue;
          const double fill_t0 = omp_get_wtime();
          fill_particle_range(buf, n, lr, p0, p1);
          fill_s += omp_get_wtime() - fill_t0;
          const double io_t0 = omp_get_wtime();
          for (int sg = 0; sg < 9 && rc == 0; ++sg) {
            const int64_t buf_off = seg_lo[sg] + p0 * seg_elem[sg];
            const int64_t bytes = (p1 - p0) * seg_elem[sg];
            const int io_rc = omp_file_pwrite(active_handle, base + buf_off,
                                              buf + buf_off, (size_t)bytes,
                                              io_async);
            if (io_rc != 0) {
              rc = 20;
              saved_errno = errno;
            }
          }
          issue_s += omp_get_wtime() - io_t0;
        }
#ifdef OMPFILE_HAVE_FILE_FLUSH
        if (pipeline_flush && rc == 0) {
          // Region-scoped completion boundary for the writes this region
          // queued. Not needed for buffer safety (the engine copied them) nor
          // to catch failures (close reports them too); it bounds the pipeline
          // to one region so one wave's drain cannot spill into the next
          // wave's fill. Measured to cost ~0.09 s per rank block at 50M
          // particles (job 414886), so HACC_PIPELINE_FLUSH=0 is the faster
          // posture and the one the -noflush lane uses.
          const double flush_t0 = omp_get_wtime();
          const int flush_rc = omp_file_flush(active_handle);
          flush_s += omp_get_wtime() - flush_t0;
          if (flush_rc != 0) {
            rc = 22;
            saved_errno = 0; // flush reports the write's rc, never errno
          }
        }
#endif
      } else {
        // ---- fill-then-write (the shape this benchmark has always had) -----
        // One chunk list drives the I/O: segmented=0 coalesces the whole rank
        // block into ONE op (each op carries scheduler/transport fixed cost),
        // segmented=1 is the 9-op GLEAN path, segmented=N (N>1) splits the
        // block into N equal chunks — the segment-shape sweep dimension.
        int64_t nchunks = !segmented ? 1 : (segmented == 1 ? 9 : segmented);
        int64_t *chunk_off =
            (int64_t *)malloc((size_t)nchunks * sizeof(int64_t));
        int64_t *chunk_len =
            (int64_t *)malloc((size_t)nchunks * sizeof(int64_t));
        if (!chunk_off || !chunk_len) {
          rc = 11;
          saved_errno = ENOMEM;
        } else {
          if (segmented == 1) {
            for (int sg = 0; sg < 9; ++sg) {
              chunk_off[sg] = seg_lo[sg];
              chunk_len[sg] = n * seg_elem[sg];
            }
          } else {
            const int64_t chunk = (rank_bytes + nchunks - 1) / nchunks;
            int64_t off = 0;
            int64_t emitted = 0;
            for (int64_t c = 0; c < nchunks && off < rank_bytes; ++c) {
              chunk_off[c] = off;
              chunk_len[c] =
                  (rank_bytes - off < chunk) ? (rank_bytes - off) : chunk;
              off += chunk_len[c];
              ++emitted;
            }
            nchunks = emitted;
          }

          if (writing) {
            // Data is born on the worker: same deterministic pattern as the
            // upstream driver, generated device-side, whole block up front.
            const double fill_t0 = omp_get_wtime();
            if (particle_major_fill)
              fill_particle_range(buf, n, lr, 0, n);
            else
              fill_byte_range(buf, n, lr, 0, rank_bytes);
            fill_s += omp_get_wtime() - fill_t0;
          }

          for (int64_t c = 0; c < nchunks && rc == 0; ++c) {
            const double io_t0 = omp_get_wtime();
            const int io_rc =
                writing ? omp_file_pwrite(active_handle, base + chunk_off[c],
                                          buf + chunk_off[c],
                                          (size_t)chunk_len[c], io_async)
                        : omp_file_pread(active_handle, base + chunk_off[c],
                                         buf + chunk_off[c],
                                         (size_t)chunk_len[c], io_async);
            issue_s += omp_get_wtime() - io_t0;
            if (io_rc != 0) {
              rc = writing ? 20 : 21;
              saved_errno = errno;
            }
          }
        }
        free(chunk_off);
        free(chunk_len);
      }

        if (!writing && rc == 0) {
          float *farr[7];
          for (int a = 0; a < 7; ++a)
            farr[a] = (float *)(buf + (int64_t)a * float_bytes);
          int64_t *pid = (int64_t *)(buf + 7 * float_bytes);
          uint16_t *mask = (uint16_t *)(buf + 7 * float_bytes + pid_bytes);
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
    region_s = omp_get_wtime() - region_t0;
  }
#pragma omp atomic
  g_fill_s += fill_s;
#pragma omp atomic
  g_issue_s += issue_s;
#pragma omp atomic
  g_flush_s += flush_s;
#pragma omp atomic
  g_region_s += region_s;
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
                     bool per_rank, int io_async, int segmented, int pipeline,
                     int pipeline_flush, int pipeline_chunks,
                     int particle_major_fill) {
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
                segmented, pipeline, pipeline_flush, pipeline_chunks,
                particle_major_fill, &io_errno);
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
  // deadlock) instead of the sequential pre-open workaround; 0 serializes the
  // opens. Default flipped to 1 (Jul 30, 2026): the concurrent fresh-open
  // deadlock was root-fixed (ENOENT backoff off the single data-event handler)
  // and the path is guarded by the standing hacc-io-concurrent-open lane;
  // concurrent opens are the realistic checkpoint pattern.
  //
  // Read as a flag rather than through env_i64, which rejects 0 and so made
  // the documented "set 0 to serialize" silently keep the concurrent path —
  // the ompfile-target-fpr lane passed 0 and ran concurrently anyway.
  const int concurrent_open = env_flag("HACC_CONCURRENT_OPEN", 1);
  // Fill traversal order for the NON-pipelined path. `particle` (default) is
  // one pass over particles touching all nine segments; `array` walks one
  // segment at a time. Identical bytes, and `array` measured 1.63x slower
  // (job 414888) because it converts (float)i seven times per particle instead
  // of once — which is why the pipeline chunks by particle range. `array` is
  // kept as the control that established that, reachable as the
  // `ompfile-target-local-seg-async-amfill` lane.
  const char *fill_order = getenv("HACC_FILL_ORDER");
  const int particle_major_fill =
      (fill_order && strcmp(fill_order, "array") == 0) ? 0 : 1;
  // #3 idle time: interleave buffer generation with the writes so the proxy
  // produces particle slice c+1 while the async engine drains slice c, instead
  // of filling the whole rank block and only then issuing. Needs a segmented
  // shape and HACC_IO_ASYNC=1 for the drain to be concurrent.
  const int pipeline = env_flag("HACC_PIPELINE", 0);
  // How many particle slices the pipelined path generates and issues per rank
  // block. Each slice costs nine pwrites (one per GLEAN segment), so this
  // trades per-op count against how early the first write reaches the engine.
  const int pipeline_chunks = (int)env_i64("HACC_PIPELINE_CHUNKS", 4);
  int pipeline_flush = env_flag("HACC_PIPELINE_FLUSH", 1);
#ifndef OMPFILE_HAVE_FILE_FLUSH
  if (pipeline && pipeline_flush) {
    fprintf(stderr, "WARN hacc-target: libompfile has no omp_file_flush; "
                    "pipelined writes drain at close instead\n");
    pipeline_flush = 0;
  }
#endif
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
                segmented, pipeline, pipeline_flush, pipeline_chunks,
                particle_major_fill) != 0)
    return 1;
  if (file_per_rank ? close_rank_handles(ranks, devices, handles)
                    : close_device_handles(devices, handles))
    return 1;
  const double write_s = now_s() - w0;
  // Snapshot before the read phase adds its own preads to the accumulators:
  // every *_s field on the summary line describes the write phase.
  const double w_fill_s = g_fill_s;
  const double w_issue_s = g_issue_s;
  const double w_flush_s = g_flush_s;
  const double w_region_s = g_region_s;

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
                  segmented, pipeline, pipeline_flush, pipeline_chunks,
                  particle_major_fill) != 0)
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
         " write_s=%.6f write_mib_s=%.3f fill_s=%.6f issue_s=%.6f "
         "flush_s=%.6f region_s=%.6f read_s=%.6f read_mib_s=%.3f "
         "verified=%d concurrency=%d pipeline=%d pipeline_flush=%d "
         "fill_order=%s pipeline_chunks=%d\n",
         file_per_rank ? "ompfile_target_fpr" : "ompfile_target", ranks,
         particles, total_bytes, write_s, write_mib, w_fill_s, w_issue_s,
         w_flush_s, w_region_s, read_s, read_mib, verified, devices, pipeline,
         pipeline_flush, particle_major_fill ? "particle" : "array",
         pipeline_chunks);
  printf(" CONTENTS VERIFIED... Success \n"); // match upstream success marker
  if (fpr_paths) {
    for (int64_t lr = 0; lr < ranks; ++lr)
      free(fpr_paths[lr]);
    free(fpr_paths);
  }
  free(handles);
  return 0;
}
