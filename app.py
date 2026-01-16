# app.py
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from inference import DamageInferenceModel

# -------------------------
# STREAMLIT CONFIG
# -------------------------
st.set_page_config(
    page_title="Car Damage Detection",
    layout="centered"
)

# -------------------------
# DEVICE SELECTION
# -------------------------
use_cuda = st.checkbox("Use GPU (CUDA)", value=False)
DEVICE = "cuda" if use_cuda else "cpu"

# -------------------------
# LOAD MODEL (CACHED PER DEVICE)
# -------------------------
@st.cache_resource
def load_engine(device):
    return DamageInferenceModel(device=device)

engine = load_engine(DEVICE)

# -------------------------
# UI
# -------------------------
st.title("🚗 Car Damage Detection & Captioning")

uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Inference"):
        with st.spinner("Analyzing image..."):
            result = engine(image)

        # -------------------------
        # VISUALIZATION
        # -------------------------
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)

        if result["damaged"]:

            # LOCATION BOX
            if result["location_box"] is not None:
                x1, y1, x2, y2 = result["location_box"].astype(int)

                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none"
                )
                ax.add_patch(rect)

                ax.text(
                    x1,
                    max(y1 - 10, 0),
                    result["location"],
                    color="white",
                    fontsize=10,
                    bbox=dict(facecolor="red", alpha=0.7)
                )

            # SEVERITY BOX (optional)
            if result.get("severity_box") is not None:
                sx1, sy1, sx2, sy2 = result["severity_box"].astype(int)

                rect = patches.Rectangle(
                    (sx1, sy1),
                    sx2 - sx1,
                    sy2 - sy1,
                    linewidth=2,
                    edgecolor="blue",
                    facecolor="none"
                )
                ax.add_patch(rect)

        ax.axis("off")
        st.pyplot(fig)

        # -------------------------
        # RESULTS
        # -------------------------
        st.subheader("Results")

        st.write(f"**Damaged:** `{result['damaged']}`")
        st.write(f"**Damage Probability:** `{result['damage_probability']:.2f}`")

        if result["severity"] is not None:
            st.write(f"**Severity:** `{result['severity']}`")

        st.write(f"**Caption:** {result['caption']}")
