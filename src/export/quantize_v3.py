"""
PhotoSense - Static INT8 Quantization V3
Two improvements aimed at EfficientNet:
  1. quant_pre_process: fold consts, optimize graph BEFORE quant
  2. per_channel: per-channel weights quant (better for Conv)
"""

import sys
from pathlib import Path
import numpy as np
from onnxruntime.quantization import (
    quantize_static, QuantType, CalibrationDataReader, QuantFormat,
    shape_inference,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.train.dataset import PhotoSenseDataset

FP32_PATH = Path("models/photosense_b3_fp32.onnx")
FP32_PREPROCESSED = Path("models/photosense_b3_fp32_pre.onnx")
INT8_V3_PATH = Path("models/photosense_b3_int8_v3.onnx")
CALIBRATION_SIZE = 500


class PhotoSenseCalibrationReader(CalibrationDataReader):
    def __init__(self, n_samples=500, batch_size=8):
        train_ds = PhotoSenseDataset("train")
        indices = list(range(min(n_samples, len(train_ds))))
        subset = [(train_ds[i][0], train_ds[i][1]) for i in indices]
        self.batches = []
        for i in range(0, len(subset), batch_size):
            batch = subset[i:i + batch_size]
            imgs = np.stack([s[0].numpy() for s in batch])
            self.batches.append({"input": imgs})
        self.iter = iter(self.batches)
        print(f"  Calibration: {n_samples} samples in {len(self.batches)} batches")

    def get_next(self):
        return next(self.iter, None)


def quantize():
    print("=" * 70)
    print("PhotoSense - Static Quantization V3 (preprocess + per_channel)")
    print("=" * 70)

    print(f"\n[1/4] Loading FP32 model: {FP32_PATH}")
    fp32_size = FP32_PATH.stat().st_size / 1024 / 1024
    print(f"  FP32 size: {fp32_size:.2f} MB")

    print(f"\n[2/4] Pre-processing model (shape inference + const folding)")
    shape_inference.quant_pre_process(
        input_model=str(FP32_PATH),
        output_model_path=str(FP32_PREPROCESSED),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
        auto_merge=True,
        int_max=2**31 - 1,
        guess_output_rank=False,
        verbose=0,
    )
    print(f"  Preprocessed: {FP32_PREPROCESSED.stat().st_size/1024/1024:.2f} MB")

    print(f"\n[3/4] Calibration ({CALIBRATION_SIZE} samples)")
    reader = PhotoSenseCalibrationReader(n_samples=CALIBRATION_SIZE, batch_size=8)

    print(f"\n[4/4] Running quantize_static (per_channel=True)...")
    quantize_static(
        model_input=str(FP32_PREPROCESSED),
        model_output=str(INT8_V3_PATH),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    int8_size = INT8_V3_PATH.stat().st_size / 1024 / 1024
    print(f"\n  INT8 V3 size: {int8_size:.2f} MB")
    print(f"  Compression: {fp32_size:.2f} -> {int8_size:.2f} MB ({(1-int8_size/fp32_size)*100:.1f}% reduction)")

    print("\n" + "=" * 70)
    print(f"V3 Done! File: {INT8_V3_PATH}")
    print(f"Next: evaluate with evaluate_v3.py")
    print("=" * 70)


if __name__ == "__main__":
    quantize()
