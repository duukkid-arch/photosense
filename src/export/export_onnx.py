"""
PhotoSense - Export PyTorch model to ONNX (legacy stable mode)
"""

import sys
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.train.model import create_model

CHECKPOINT = Path("models/best_efficientnet_b3.pt")
ONNX_PATH = Path("models/photosense_b3_fp32.onnx")


def export_to_onnx():
    print("=" * 70)
    print("PhotoSense -> ONNX Export (legacy)")
    print("=" * 70)

    print(f"\n[1/4] Loading PyTorch model: {CHECKPOINT}")
    device = torch.device("cpu")
    model = create_model("efficientnet_b3", num_classes=24, pretrained=False).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  From epoch {ckpt['epoch']}, val_acc = {ckpt['val_acc']:.4f}")

    pytorch_size_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
    print(f"  PyTorch param size: {pytorch_size_mb:.2f} MB (expected ONNX size)")

    print("\n[2/4] Preparing dummy input (1, 3, 224, 224)")
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"\n[3/4] Exporting to ONNX (using dynamo=False, legacy TorchScript) -> {ONNX_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )
    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"  ONNX file size: {size_mb:.2f} MB")

    if size_mb < 5:
        print(f"  WARNING: File too small! Weights may not be embedded.")
    else:
        print(f"  OK: File size matches expected.")

    print("\n[4/4] Verifying ONNX model")
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("  ONNX model structure: OK")

    print("\n[Compare inference results]")
    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()
    ort_session = onnxruntime.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_output = ort_session.run(None, {"input": dummy_input.numpy()})[0]
    diff = np.abs(pytorch_output - onnx_output).max()
    print(f"  Max numerical diff: {diff:.6f}")
    print(f"  {'OK: outputs match' if diff < 1e-4 else 'WARN: large diff'}")

    print("\n" + "=" * 70)
    print(f"ONNX Export Done! File size: {size_mb:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    export_to_onnx()
