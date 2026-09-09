# Benchmark Methodology

Performance results in this project must come from measured execution rather than estimated values.

## Metrics

The benchmark suite records:

- requests
- concurrency
- wall-clock duration
- throughput
- mean latency
- p50 latency
- p95 latency
- p99 latency

Two latency boundaries are measured independently.

---

## Triton Request Latency

Measures:

```text
client
  ↓
HTTP
  ↓
Triton
  ↓
TensorRT
  ↓
response
```

This is useful for analyzing model-serving performance.

---

## End-to-End Frame Latency

Measures:

```text
decoded RGB frame
      ↓
MONAI preprocessing
      ↓
Triton request
      ↓
TensorRT
      ↓
response processing
      ↓
temporal aggregation
```

This better represents application-visible inference latency.

Video decode and disk I/O are deliberately excluded.

---

## TensorRT Baseline

Run batch-size 1:

```bash
python scripts/benchmark_tensorrt.py \
  --engine models/surgical_phase.plan \
  --batch-size 1 \
  --output benchmarks/results/tensorrt_batch1.json
```

Additional batches may be measured with:

```text
1
2
4
8
16
32
```

---

## Streaming Concurrency Sweep

Recommended runs:

```bash
python scripts/benchmark_streaming.py \
  --video /path/to/video41.mp4 \
  --concurrency 1 \
  --requests-per-stream 100 \
  --output benchmarks/results/stream_c1.json
```

```bash
python scripts/benchmark_streaming.py \
  --video /path/to/video41.mp4 \
  --concurrency 2 \
  --requests-per-stream 100 \
  --output benchmarks/results/stream_c2.json
```

```bash
python scripts/benchmark_streaming.py \
  --video /path/to/video41.mp4 \
  --concurrency 4 \
  --requests-per-stream 100 \
  --output benchmarks/results/stream_c4.json
```

```bash
python scripts/benchmark_streaming.py \
  --video /path/to/video41.mp4 \
  --concurrency 8 \
  --requests-per-stream 100 \
  --output benchmarks/results/stream_c8.json
```

Continue increasing concurrency until throughput stops scaling or tail latency becomes unacceptable.

---

## Triton Performance Analyzer

Run:

```bash
./scripts/run_perf_analyzer.sh
```

This provides an independent Triton-specific concurrency benchmark.

The application benchmark and Performance Analyzer serve different purposes:

```text
Application benchmark
→ full preprocessing + client path

Performance Analyzer
→ serving-focused load generation
```

Both should be retained in final results.

---

## Reporting Results

Do not report only average latency.

The final benchmark table should include:

| Concurrency | Throughput | Mean | P50 | P95 | P99 |
|---:|---:|---:|---:|---:|---:|
| 1 | Pending | Pending | Pending | Pending | Pending |
| 2 | Pending | Pending | Pending | Pending | Pending |
| 4 | Pending | Pending | Pending | Pending | Pending |
| 8 | Pending | Pending | Pending | Pending | Pending |

Also record:

```text
GPU
CUDA version
TensorRT version
Triton version
model precision
batch profile
```

so results remain reproducible.

---

## Current Status

Benchmark infrastructure is implemented and unit-tested.

Actual NVIDIA GPU measurements are pending deployment to a CUDA-capable Linux system.
