# VRAM Flow

> Keep GPUs computing when the working set is larger than GPU memory.

## The Problem: GPU Memory Is Finite, But Working Sets Are Not

CUDA applications are frequently constrained not by arithmetic throughput, but by **data availability**. A GPU can 
execute kernels only on data it can access, while high-performance device memory (VRAM/HBM) has finite capacity. When an 
application's active dataset exceeds available device memory, data must be staged from another tier (peer GPU memory, 
host memory, local storage, or remote storage) before it can be consumed efficiently.

CUDA Unified Memory can oversubscribe GPU memory and migrate memory automatically, but migration may be reactive and 
page-fault-driven. GPUDirect Storage can provide a direct DMA path between storage and GPU memory, but it is a 
**data-movement mechanism**, not a policy engine that decides what data should be resident, prefetched, retained, 
replicated, or evicted. RMM provides sophisticated GPU and host memory resources, but primarily addresses allocation and 
memory-resource management rather than distributed placement of application data.

For applications whose working set exceeds aggregate VRAM, the hard problem is therefore not simply:

> **How do we store more data than fits on the GPUs?**

It is:

> **How do we continuously place the right data on the right GPU early enough that data movement is hidden behind useful 
> computation?**

If a kernel finishes and its next input is still in host memory, NVMe, or on another node, the GPU stalls regardless of 
its theoretical compute throughput. The target is not zero transfer latency (that is physically impossible) but 
**near-zero GPU idle time caused by data unavailability**, by overlapping transfers with computation whenever workload 
structure and hardware bandwidth permit.

## Proposal

**VRAM Flow** is a CUDA-native, distributed buffer cache for workloads whose working sets exceed available GPU memory.

Applications address logical buffers rather than manually orchestrating every movement between storage tiers. VRAM Flow 
tracks where each buffer is resident and coordinates **placement, prefetching, caching, eviction, and transfer** across 
a hierarchy such as:

``` text
GPU VRAM/HBM <-> Peer GPU <-> Pinned Host RAM <-> Local NVMe <-> Remote Storage
```

The data plane should exploit the fastest available path (such as asynchronous CUDA copies, peer-to-peer transfers, 
GPUDirect Storage, or RDMA where supported) while remaining functional on ordinary CUDA systems. The control plane 
maintains buffer metadata, placement decisions, leases/lifetimes, topology information, and transfer state.

A critical requirement is **CUDA stream-aware lifetime safety**. A buffer cannot be evicted merely because the host 
considers it unused: an asynchronously executing kernel may still reference it. VRAM Flow therefore associates buffer 
use with CUDA streams/events so memory becomes reclaimable only after dependent GPU work has completed.

The long-term objective is simple:

> Given a computation and a dataset larger than VRAM, keep each GPU supplied with the data it needs while minimizing 
> exposed data-movement latency.

VRAM Flow does **not** claim to make remote storage equivalent to VRAM or eliminate I/O latency. Performance remains 
bounded by storage, interconnect, topology, access predictability, and compute/transfer overlap.

## Why Existing Solutions Do Not Fully Provide This Abstraction

| Technology | What It Solves | Remaining Gap For VRAM Flow |
|-|-|-|
| **CUDA Unified Memory** | Unified addressing, migration, and supported GPU-memory oversubscription. | General memory-management mechanism. Migration can be reactive. It is not a distributed, application-level buffer-placement and prefetch service. |
| **GPUDirect Storage (GDS)** | Direct DMA path between storage and GPU memory, avoiding the traditional CPU bounce buffer. | Efficient **transport**, not cache/placement policy. It does not decide which application buffers should move, when to prefetch them, or what to evict. |
| **RAPIDS Memory Manager (RMM)** | CUDA/host allocators, pools, arenas, pinned memory, and memory-resource abstractions. | Excellent allocation layer, but not a general distributed object/buffer cache coordinating data residency across machines and storage tiers. |
| **Dask-CUDA / RAPIDS** | GPU-aware distributed execution and memory spilling for the RAPIDS/Dask ecosystem. | Strong solution inside its execution/dataframe ecosystem. VRAM Flow targets a lower-level, framework-independent C++/CUDA buffer abstraction usable by arbitrary CUDA applications. |
| **Mooncake and GPU/KV-cache systems** | High-performance distributed transfer/storage, especially for AI/LLM serving and KV-cache workloads. | Demonstrate the value of disaggregated GPU-aware storage, but are oriented toward AI-serving architectures rather than a general-purpose CUDA buffer-lifetime and placement API. |

VRAM Flow should **compose with these technologies rather than reimplement them**. GDS can become a transfer backend, 
RMM can provide allocators, RDMA can provide a network fast path. VRAM Flow's intended contribution is the missing 
orchestration layer between an application's logical buffers and those mechanisms.

## Closing

Modern GPUs are extraordinarily fast when their operands are ready. Once the working set exceeds device memory, however, 
keeping those operands ready becomes a systems problem spanning CUDA execution semantics, memory hierarchy, storage, 
networking, caching, scheduling, and failure handling.

VRAM Flow aims to make that problem explicit and reusable.

The initial milestone should be deliberately narrow: **one node, multiple CUDA GPUs, immutable buffers, pinned-host/NVMe 
spill, asynchronous prefetch, topology-aware placement, and stream/event-safe eviction**. Distributed metadata, remote 
storage, RDMA, replication, and failure recovery can then extend the same abstraction across machines.

The project's success criterion is not "support datasets larger than VRAM", CUDA ecosystems already provide ways to do 
that.

**It is keeping GPUs busy when the data they need does not all fit in VRAM.**
