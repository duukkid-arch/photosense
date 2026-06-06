"""
PhotoSense - Static INT8 Quantization with calibration data

Static quantization uses real data to compute activation scales,
giving much better accuracy than dynamic quantization on EfficientNet.
"""

import sys
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.train.dataset import PhotoSenseDataset

FP32_PATH = Path("models/photosense_b3_fp32.onnx")
INT8_STATIC_PATH = Path("models/photosense_b3_int8_static.onnx")
CALIBRATION_SIZE = 200  # 200 张 train 图做 calibration


class PhotoSenseCalibrationReader(CalibrationDataReader):
    """
    Feeds calibration images to the quantizer.
    Reuses our PhotoSense train dataset for real-world activations.
    """
    def __init__(self, n_samples=200, batch_size=8):
        train_ds = PhotoSenseDataset("train")
        # Take first n_samples for calibration
        indices = list(range(min(n_samples, len(train_ds))))
        subset = [(train_ds[i][0], train_ds[i][1]) for i in indices]
        self.batch_size = batch_size
        self.batches = []
        for i in range(0, len(subset), batch_size):
            batch = subset[i:i + batch_size]
            imgs = np.stack([s[0].numpy() for s in batch])
            self.batches.append({"input": imgs})
        self.iter = iter(self.batches)
        print(f"  Calibration: {n_samples} samples, {len(self.batches)} batches")

    def get_next(self):
        return next(self.iter, None)


def quantize():
    print("=" * 70)
    print("PhotoSense - Static INT8 Quantization")
    print("=" * 70)

    print(f"\n[1/3] Loading FP32 model: {FP32_PATH}")
    fp32_size = FP32_PATH.stat().st_size / 1024 / 1024
    print(f"  FP32 size: {fp32_size:.2f} MB")

    print(f"\n[2/3] Preparing calibration data reader ({CALIBRATION_SIZE} train samples)")
    reader = PhotoSenseCalibrationReader(n_samples=CALIBRATION_SIZE, batch_size=8)

    print(f"\n[3/3] Running static quantization (this takes 2-5 minutes)...")
    quantize_static(
        model_input=str(FP32_PATH),
        model_output=str(INT8_STATIC_PATH),
        calibration_data_reader=reader,
        quant_format=2,  # QDQ format (portable, Windows compatible)
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    int8_size = INT8_STATIC_PATH.stat().st_size / 1024 / 1024
    print(f"\n  INT8 (static) size: {int8_size:.2f} MB")
    print(f"  Compression: {fp32_size:.2f} -> {int8_size:.2f} MB ({(1-int8_size/fp32_size)*100:.1f}% reduction)")

    print("\n" + "=" * 70)
    print(f"Static Quantization Done!")
    print(f"  File: {INT8_STATIC_PATH}")
    print(f"  Next: run evaluate_onnx.py to check accuracy")
    print("=" * 70)


if __name__ == "__main__":
    quantize()
