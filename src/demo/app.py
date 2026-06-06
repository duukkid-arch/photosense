"""
PhotoSense - Streamlit Demo
"""

import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.train.model import create_model
from src.dataset.categories import CATEGORY_LIST, CATEGORIES

CHECKPOINT = Path("models/best_efficientnet_b3.pt")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = create_model("efficientnet_b3", num_classes=24, pretrained=False).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["val_acc"], ckpt["epoch"]


def predict(model, image):
    img_t = PREPROCESS(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)[0]
    top5 = torch.topk(probs, k=5)
    results = []
    for prob, idx in zip(top5.values, top5.indices):
        cat_key = CATEGORY_LIST[idx.item()]
        cat_cn = CATEGORIES[cat_key]["cn"]
        results.append({"key": cat_key, "cn": cat_cn, "prob": prob.item()})
    return results


st.set_page_config(page_title="PhotoSense", page_icon=":camera:", layout="wide")

st.title("PhotoSense")
st.caption("24-class photographic scene classifier | EfficientNet-B3 | Test acc 82.01%")

with st.sidebar:
    st.header("About")
    st.write("**Model:** EfficientNet-B3")
    st.write("**Classes:** 24 photographic scenes")
    st.write("**Test Accuracy:** 82.01%")
    st.write("**Training Data:** 14,400 Unsplash images")
    st.write("**GitHub:** [duukkid-arch/photosense](https://github.com/duukkid-arch/photosense)")
    st.markdown("---")
    st.subheader("Supported scenes")
    st.write("portrait_indoor, portrait_outdoor, portrait_night, selfie")
    st.write("landscape_mountain, landscape_sea, landscape_sky, sunset_sunrise")
    st.write("city_skyline, architecture, street, night_city")
    st.write("food_chinese, food_western, dessert, drink")
    st.write("pet_cat, pet_dog, flower, indoor_home")
    st.write("product, sports, concert, rain")

model, val_acc, epoch = load_model()
st.success(f"Model ready (epoch {epoch}, val_acc {val_acc:.4f})")

uploaded = st.file_uploader("Upload a photo to classify", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Your photo", use_container_width=True)

    with col2:
        st.subheader("Classification Results")
        with st.spinner("Analyzing..."):
            results = predict(model, image)

        top1 = results[0]
        st.markdown(f"### Top prediction: **{top1['cn']}**")
        st.code(top1["key"], language=None)
        st.progress(top1["prob"])
        st.caption(f"Confidence: {top1['prob']*100:.1f}%")

        st.markdown("---")
        st.subheader("Top-5 candidates")
        for i, r in enumerate(results):
            st.markdown(f"**{i+1}. {r['cn']}** ({r['key']})")
            st.progress(r["prob"])
            st.caption(f"{r['prob']*100:.2f}%")
else:
    st.info("Upload a photo to see classification.")
    st.markdown("Try with any image from your computer or grab one from data/raw/<category>/")
