# Real-Time Surgical Video Inference

A real-time surgical phase recognition pipeline built with MONAI,
ONNX, NVIDIA TensorRT, and NVIDIA Triton Inference Server.

The project uses laparoscopic surgical video to classify the current
phase of a cholecystectomy procedure and benchmarks the complete
inference path from PyTorch to optimized TensorRT deployment.

## Planned Pipeline

Video Stream
→ Frame/Clip Processing
→ MONAI Model
→ ONNX
→ TensorRT
→ Triton Inference Server
→ Streaming Client

## Status

Initial project setup.