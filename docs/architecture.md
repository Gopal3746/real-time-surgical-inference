# System Architecture

## Overview

The surgical phase recognition system is organized into four layers:

```text
Data
  ↓
Model
  ↓
Deployment
  ↓
Serving + Performance
```

Each layer is designed to be independently testable.

---

## 1. Data Layer

```text
Cholec80 video
      +
phase annotation
      │
      ▼
manifest builder
      │
      ▼
video-level train / val / test split
      │
      ▼
temporal frame sampling
```

The split occurs at the video level to prevent correlated frames from the same surgery from appearing across datasets.

---

## 2. Model Layer

```text
RGB frame
    │
    ▼
MONAI preprocessing
    │
    ├── channel-first
    ├── intensity scaling
    ├── resize
    └── normalization
    │
    ▼
3 × 224 × 224
    │
    ▼
EfficientNet-B0
    │
    ▼
7 phase logits
```

The inference contract is:

```text
Input:
N × 3 × 224 × 224 FP32

Output:
N × 7 FP32
```

---

## 3. Temporal Layer

Surgical phases normally persist across many consecutive video frames.

Instead of treating every prediction independently, logits are accumulated in a bounded temporal window:

```text
L(t-4)
L(t-3)
L(t-2)
L(t-1)
L(t)
   │
   ▼
mean logits
   │
   ▼
softmax
   │
   ▼
phase
```

This produces both:

- raw frame predictions
- temporally smoothed predictions

---

## 4. Model Export

```text
PyTorch
   │
   ▼
ONNX
   │
   ├── graph validation
   └── ONNX Runtime parity
   │
   ▼
TensorRT
```

The ONNX model supports a dynamic batch dimension.

The TensorRT deployment profile is designed for:

```text
minimum batch = 1
optimal batch = 8
maximum batch = 32
```

---

## 5. Triton Serving

```text
Client
   │
   ▼
HTTP request
   │
   ▼
Triton
   │
   ├── dynamic batcher
   │
   ▼
TensorRT plan
   │
   ▼
7 logits
```

Repository layout:

```text
model_repository/
└── surgical_phase/
    ├── config.pbtxt
    └── 1/
        └── model.plan
```

Triton owns the batch dimension.

Therefore the model configuration declares:

```text
input dims  = [3, 224, 224]
output dims = [7]
```

while the complete runtime shapes remain:

```text
[N, 3, 224, 224]
[N, 7]
```

---

## 6. Streaming Path

```text
Video
  │
  ▼
OpenCV
  │
  ▼
sampled RGB frame
  │
  ▼
MONAI
  │
  ▼
Triton client
  │
  ▼
TensorRT
  │
  ▼
logits
  │
  ▼
temporal aggregator
  │
  ▼
surgical phase
```

Each output contains:

```text
frame number
timestamp
raw phase
raw confidence
smoothed phase
smoothed confidence
Triton request latency
```

---

## 7. Concurrency Model

Each simulated surgical stream owns:

```text
one Triton client
one temporal aggregator
one preprocessing transform
```

Conceptually:

```text
Stream 1 → Client 1 ┐
Stream 2 → Client 2 │
Stream 3 → Client 3 ├── Triton
Stream 4 → Client 4 │
                    ┘
```

This avoids sharing a mutable client object between benchmark workers.

---

## 8. Benchmark Boundaries

### Triton request latency

Measured around:

```text
client.infer(...)
```

It includes the request/response serving path.

### End-to-end latency

Measured around:

```text
frame
→ preprocessing
→ inference
→ response conversion
→ temporal aggregation
```

Video decoding is intentionally excluded from this benchmark.

The benchmark frame pool is decoded before timing begins.

---

## 9. Platform Separation

```text
                     ┌─────────────────┐
                     │     macOS       │
                     │                 │
                     │ development     │
                     │ unit testing    │
                     │ MPS             │
                     │ ONNX Runtime    │
                     └────────┬────────┘
                              │
                           ONNX
                              │
                              ▼
                     ┌─────────────────┐
                     │ NVIDIA Linux    │
                     │                 │
                     │ CUDA            │
                     │ TensorRT        │
                     │ Triton          │
                     │ benchmarking    │
                     └─────────────────┘
```

The platform-specific boundary occurs after ONNX export.

This keeps most of the system hardware-independent while reserving accelerator-specific optimization for NVIDIA deployment.
