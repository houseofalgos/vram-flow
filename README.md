# VRAM Flow

**Keep GPUs computing when the working set is larger than GPU memory.**

VRAM Flow is an open-source project exploring a CUDA-native, distributed buffer management layer for workloads whose 
data exceeds available GPU memory.

The goal is to coordinate **placement, prefetching, caching, eviction, and data transfer** across GPU memory, host 
memory, local storage, and remote storage—while overlapping data movement with computation to minimize GPU idle time.

## Status

🚧 **Early development.**

VRAM Flow is currently in the design and prototyping stage. APIs, architecture, and implementation details are expected 
to change.

## Documentation

Start here:

**[Problem & Vision](docs/01_PROBLEM_AND_VISION.md)** — the problem VRAM Flow aims to solve, the proposed approach, and 
how it differs from existing CUDA memory and data-movement technologies.

## License

See [LICENSE](LICENSE).
