"""
PhotoSense - Evaluate all ONNX models on test set
"""

import sys
from pathlib import Path
import numpy as np
import onnxruntime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.train.dataset import PhotoSenseDataset
from torch.utils.data import DataLoader

FP32_PATH = Path("models/photosense_b3_fp32.onnx")
INT8_DYN_PATH = Path("models/photosense_b3_int8.onnx")
INT8_STATIC_PATH = Path("models/photosense_b3_int8_static.onnx")


def evaluate(onnx_path, test_dl):
    if not onnx_path.exists():
        return None
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    correct = 0
    total = 0
    pbar = tqdm(test_dl, desc=f"{onnx_path.name}", ncols=80)
    for imgs, labels in pbar:
        imgs_np = imgs.numpy()
        labels_np = labels.numpy()
        outputs = session.run(None, {"input": imgs_np})[0]
        preds = outputs.argmax(axis=1)
        correct += (preds == labels_np).sum()
        total += len(labels_np)
    return correct / total


def main():
    print("=" * 70)
    print("PhotoSense - All ONNX Models Evaluation")
    print("=" * 70)

    print("\n[Loading test set]")
    test_ds = PhotoSenseDataset("test")
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"  Test samples: {len(test_ds)}")

    print("\n[Evaluating FP32 ONNX]")
    fp32_acc = evaluate(FP32_PATH, test_dl)

    print("\n[Evaluating INT8 (Dynamic) ONNX]")
    int8_dyn_acc = evaluate(INT8_DYN_PATH, test_dl)

    print("\n[Evaluating INT8 (Static) ONNX]")
    int8_static_acc = evaluate(INT8_STATIC_PATH, test_dl)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  {'Model':<30} {'Acc':<10} {'Size':<10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'PyTorch .pt (baseline)':<30} {'82.01%':<10} {'40.94 MB':<10}")
    print(f"  {'FP32 ONNX':<30} {fp32_acc*100:.2f}%     {FP32_PATH.stat().st_size/1024/1024:.2f} MB")
    print(f"  {'INT8 ONNX (Dynamic)':<30} {int8_dyn_acc*100:.2f}%     {INT8_DYN_PATH.stat().st_size/1024/1024:.2f} MB")
    if int8_static_acc is not None:
        print(f"  {'INT8 ONNX (Static)':<30} {int8_static_acc*100:.2f}%     {INT8_STATIC_PATH.stat().st_size/1024/1024:.2f} MB")
        print()
        print(f"  Static Quant accuracy retention: {int8_static_acc/fp32_acc*100:.1f}% of FP32")
        print(f"  Static Quant accuracy drop:      {(fp32_acc - int8_static_acc)*100:.2f} pp")
    print("=" * 70)


if __name__ == "__main__":
    main()
