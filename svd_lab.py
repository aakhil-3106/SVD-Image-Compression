import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="SVD Image Lab", layout="wide")
st.title("SVD Image Lab")


# ---------------------------------------------------------------- core maths
@st.cache_data
def load_and_decompose(file_bytes, size):
    # Open the image and composite any transparency onto white
    img = Image.open(io.BytesIO(file_bytes)).convert("RGBA")
    img = Image.alpha_composite(Image.new("RGBA", img.size, "white"), img).convert("RGB")

    # Resize so the longest edge is at most size (never upscales)
    img.thumbnail((size, size))
    pixels = np.asarray(img, dtype=np.float64)

    # One SVD per colour channel
    svds = [np.linalg.svd(pixels[:, :, c], full_matrices=False) for c in range(3)]
    return pixels.astype(np.uint8), svds


def rebuild(svds, k):
    # Keep only the first k singular values: A_k = U_k Sigma_k V_k^T
    channels = [(U[:, :k] * S[:k]) @ Vt[:k] for U, S, Vt in svds]
    return np.clip(np.rint(np.stack(channels, axis=2)), 0, 255).astype(np.uint8)


def adjust_intensity(img, red, green, blue):
    # Scale each colour channel by its percentage (100 = unchanged)
    scale = np.array([red, green, blue]) / 100.0
    return np.clip(img.astype(np.float64) * scale, 0, 255).astype(np.uint8)


# ------------------------------------------------------------- compress
def compress(svds, k):
    # Store only the first k components of every channel (U_k, S_k, Vt_k).
    U = np.stack([u[:, :k] for u, _, _ in svds]).astype(np.float16)
    S = np.stack([s[:k] for _, s, _ in svds]).astype(np.float32)
    Vt = np.stack([vt[:k] for _, _, vt in svds]).astype(np.float16)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, U=U, S=S, Vt=Vt)
    return buffer.getvalue()


def to_png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def spectrum_chart(svds, k):
    # Combine RGB singular value strengths and normalize.
    combined = np.sqrt(sum(S**2 for _, S, _ in svds))
    combined = combined / combined[0] if combined[0] > 0 else combined

    df = pd.DataFrame({
        "index": np.arange(1, len(combined) + 1),
        "strength": combined,
        "kept": np.arange(len(combined)) < k,
    })

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("index:O", title="Singular value index", axis=alt.Axis(labelOverlap=True)),
            y=alt.Y("strength:Q", title="Relative strength"),
            color=alt.Color(
                "kept:N",
                scale=alt.Scale(domain=[True, False], range=["#245eea", "#dce4ed"]),
                legend=None,
            ),
            tooltip=["index", alt.Tooltip("strength:Q", format=".3f")],
        )
        .properties(height=180)
    )


def kb(n_bytes):
    return f"{n_bytes / 1024:.1f} KB"


# ------------------------------------------------------------------ the app
col_upload, col_size, col_k = st.columns([1, 1, 1.4])

file = col_upload.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])
size = col_size.selectbox("Processing size (longest edge)", [128, 160, 192], index=1)

if file is None:
    st.info("Upload an image to begin.")
    st.stop()

original, svds = load_and_decompose(file.getvalue(), size)
height, width, _ = original.shape
max_k = min(height, width)

k = col_k.slider("Singular values (k)", 1, max_k, min(20, max_k))

st.markdown("**Colour intensity** (100% = unchanged)")
c_red, c_green, c_blue = st.columns(3)
red = c_red.slider("Red", 0, 200, 100, step=5, format="%d%%")
green = c_green.slider("Green", 0, 200, 100, step=5, format="%d%%")
blue = c_blue.slider("Blue", 0, 200, 100, step=5, format="%d%%")

result = adjust_intensity(rebuild(svds, k), red, green, blue)

before, after = st.columns(2)
before.image(original, caption=f"Before - {width} x {height}", width="stretch")
after.image(result, caption=f"After - k = {k}", width="stretch")

compressed = compress(svds, k)
png = to_png(result)
d1, d2 = st.columns(2)
d1.download_button(
    f"Download compressed file (.npz, {kb(len(compressed))})",
    data=compressed,
    file_name=f"compressed_k{k}.npz",
    mime="application/octet-stream",
    width="stretch",
)
d2.download_button(
    f"Download image (PNG, {kb(len(png))})",
    data=png,
    file_name=f"reconstructed_k{k}.png",
    mime="image/png",
    width="stretch",
)
st.caption("The .npz file keeps only the first k components. Colour intensity changes affect the PNG only.")

st.subheader("Singular value spectrum")
st.altair_chart(spectrum_chart(svds, k), width="stretch")
st.caption("Blue bars are the singular values kept. RGB channels combined.")
