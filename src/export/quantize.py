"""
PhotoSense - Quantize FP32 ONNX to INT8 (QDQ format - portable)
"""

import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat

FP32_PATH = Path("models/photosense_b3_fp32.onnx")
INT8_PATH = Path("models/photosense_b3_int8.onnx")


def quantize():
    print("=" * 70)
    print("PhotoSense - ONNX FP32 -> INT8 Quantization (QDQ)")
    print("=" * 70)

    print(f"\n[1/3] Loading FP32 model: {FP32_PATH}")
    fp32_size = FP32_PATH.stat().st_size / 1024 / 1024
    print(f"  FP32 size: {fp32_size:.2f} MB")

    print(f"\n[2/3] Running dynamic quantization (QDQ format)...")
    quantize_dynamic(
        model_input=str(FP32_PATH),
        model_output=str(INT8_PATH),
        weight_type=QuantType.QUInt8,
    )
    int8_size = INT8_PATH.stat().st_size / 1024 / 1024
    print(f"  INT8 size: {int8_size:.2f} MB")
    print(f"  Compression: {fp32_size:.2f} MB -> {int8_size:.2f} MB ({int8_size/fp32_size*100:.1f}%)")
    print(f"  Size reduction: {(1 - int8_size/fp32_size)*100:.1f}%")

    print(f"\n[3/3] Inference speed comparison (100 runs)")
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    sess_fp32 = onnxruntime.InferenceSession(str(FP32_PATH), providers=["CPUExecutionProvider"])
    sess_int8 = onnxruntime.InferenceSession(str(INT8_PATH), providers=["CPUExecutionProvider"])

    for _ in range(5):
        sess_fp32.run(None, {"input": dummy_input})
        sess_int8.run(None, {"input": dummy_input})

    t = time.time()
    for _ in range(100):
        out_fp32 = sess_fp32.run(None, {"input": dummy_input})[0]
    fp32_ms = (time.time() - t) * 10

    t = time.time()
    for _ in range(100):
        out_int8 = sess_int8.run(None, {"input": dummy_input})[0]
    int8_ms = (time.time() - t) * 10

    print(f"  FP32 avg latency: {fp32_ms:.2f} ms")
    print(f"  INT8 avg latency: {int8_ms:.2f} ms")
    print(f"  Speedup: {fp32_ms/int8_ms:.2f}x")

    diff = np.abs(out_fp32 - out_int8).max()
    print(f"\n  Max output diff: {diff:.4f}")
    print(f"  Predicted class match: {out_fp32.argmax() == out_int8.argmax()}")

    print("\n" + "=" * 70)
    print(f"Quantization Done!")
    print(f"  FP32: {fp32_size:.2f} MB, {fp32_ms:.2f} ms")
    print(f"  INT8: {int8_size:.2f} MB, {int8_ms:.2f} ms")
    print(f"  Saved {(1 - int8_size/fp32_size)*100:.0f}% size, {fp32_ms/int8_ms:.1f}x faster")
    print("=" * 70)


if __name__ == "__main__":
    quantize()
