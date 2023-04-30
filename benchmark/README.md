# Benchmark

This folder contains simple benchmarks of the CTGTT library.

## Build 

First make sure to build all dependencies in `src/external`:

```bash 
# BLIS
src/external/blis $ ./configure auto
src/external/blis $ make

# HPTT
src/external/hptt $ make

# Marray
src/external/marray $ ./configure
src/external/marray $ make
```

Depending on your architecture, it might be necessary to update the `INCLUDE` variable of BLIS in the Makefile.

After that build the benchmarks with `make`.
