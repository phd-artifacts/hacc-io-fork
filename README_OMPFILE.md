# HACC-IO fork with an OMPFILE (libompfile + MPP) variant

Vendored from the CORAL HACC_IO checkpoint/restart kernel (Argonne GLEAN,
BSD-style license — see LICENSE). Upstream mirror:
github.com/glennklockwood/hacc-io @ `b4d8c1d` (includes the printinfo PR).
This directory is a standalone git repository, intended to be tracked as a
submodule of io-playground (like `application/ior-fork`).

Upstream files are untouched (`restartio_glean.h`,
`glean_util_printmessage.h`, `Makefile`, `testhacc_open_close.cc`,
`testhacc_printinfo.cc`) except one hardening change in
`restartio_glean.cc`: the three `MPI_File_open` calls checked only by bare
`assert` (compiled out under Release, so an open failure surfaced as a
cascade of "Invalid file handle" writes) now report via
`__HandleMPIIOError` and `MPI_Abort`. Local additions/changes:

- `testhacc_io.cc` (instrumented upstream driver, MPI SPMD): interface
  selection via `HACC_IO_INTERFACE` (`posix` read/write | `posix-pwrite` |
  `mpiio`), barrier-fenced write/read phase timing, and a machine-parseable
  `HACC_IO_SUMMARY` line on rank 0 (aggregate bytes, write/read s + MiB/s,
  verified flag). Baselines exercise the pristine `RestartIO_GLEAN` class.
- `testhacc_ompfile.cc` (new, MPI-free): split-role OMPFILE variant. Runs as
  the single app rank of an MPP launch (proxies on ranks 0..N-2); emulates
  `HACC_RANKS` logical ranks writing/reading the GLEAN single-shared-file
  layout (24 MB header + per-rank contiguous blocks of the nine particle
  arrays, 38 B/particle) through `omp_file_pwrite/pread`, then verifies
  restart contents bit-exactly against the upstream fill pattern.
- `CMakeLists.txt`: builds both drivers (`testhacc_io` needs MPI CXX;
  `testhacc_ompfile` needs the repo LLVM toolchain + libompfile).
- `run_sorgan_hacc_io_compare.sbatch` / `run_amd_hacc_io_compare.sbatch`:
  four-lane backend compare in one allocation — `posix`, `mpiio` (N SPMD
  ranks), `ompfile-wt` (write-through, fsync-per-write), `ompfile-wb`
  (readthrough staging + write-back capture + fsync-policy close). Driven by
  the `hacc-io-smoke` / `hacc-io` spinner levels in
  `tools/sorgan_remote.sh` / `tools/amd_remote.sh`
  (`spinner/level-1-hacc-io/`).

Methodology (matches the IOR backend-compare lane): POSIX/MPIIO baselines
run N SPMD ranks; the OMPFILE variant runs split-role (1 app rank + N-1 MPP
proxies) with `HACC_RANKS = N` so aggregate bytes match — the
split-role-vs-direct-rank caveat from docs/quick-reference.md applies.
