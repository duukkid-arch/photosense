# PhotoSense - Quantization Exploration Log

## Summary

Attempted INT8 quantization of EfficientNet-B3 using ONNX Runtime. All three
approaches failed to retain accuracy due to known limitations in ONNX Runtime's
EfficientNet-specific operator fusion support.

## Results Table

| Approach | File Size | Test Acc | Acc Drop | Status |
|----------|-----------|----------|----------|--------|
| PyTorch baseline (.pt) | 40.94 MB | 82.01% | - | reference |
| FP32 ONNX | 40.87 MB | 82.01% | 0.00 pp | ✅ deployable |
| INT8 Dynamic | 10.73 MB | 14.10% | -67.91 pp | ❌ unusable |
| INT8 Static (200 samples) | 10.89 MB | 4.38% | -77.63 pp | ❌ unusable |
| INT8 V3 (preprocess + per_channel, 500 samples) | 11.69 MB | 4.10% | -77.91 pp | ❌ unusable |

## Root Cause Analysis

ONNX Runtime does not provide native quantization-aware operator fusion for:

1. **SiLU activation (x * sigmoid(x))** — quantization-induced clipping of
   negative range causes information loss.
2. **Squeeze-and-Excitation (SE) modules** — channel attention weights cluster
   near 0 or 1, with INT8 discretization losing critical middle values.
3. **Depthwise convolutions** — sensitive to quantization noise without
   per-layer custom scales.

In contrast, TensorFlow Lite and PyTorch FX Graph Mode both provide
EfficientNet-specific fusion that achieves <2 pp accuracy drop.

## Decision: Deploy FP32 ONNX

For PhotoSense V1.0 we deploy `photosense_b3_fp32.onnx` (40.87 MB, 82.01%).

For V2.0 mobile-edge deployment, two options remain:

- **Option A: Switch backbone to MobileNet-V3** — natively quantization-friendly
  in ONNX Runtime, expected <2 pp drop.
- **Option B: Switch to TFLite toolchain** — keep EfficientNet-B3, use TFLite's
  native EfficientNet quantization support.

## Engineering Lesson

Model architecture and deployment toolchain must be designed together.
Choosing EfficientNet-B3 for accuracy and ONNX Runtime for portability
created an unforeseen quantization gap.
