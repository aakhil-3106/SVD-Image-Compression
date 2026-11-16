# Image Compression Using SVD

Mathematical Foundations of AI & Data Science mini project, AIML Section D, PES University.

## Run

Requires Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python -m streamlit run svd_lab.py
```

Upload a PNG, JPEG or WebP image, choose a processing size and adjust k. The app shows the resized original beside its rank-k reconstruction, RGB intensity controls, and a singular value spectrum. Download truncated SVD factors as NPZ or the reconstructed PNG.

## Mathematics

Each RGB channel is a matrix A. NumPy computes A = U Sigma V^T. Keeping the first k components gives A_k = U[:, :k] Sigma[:k] V^T[:k, :]. Larger k preserves more detail and stores more values. Each channel uses k(m+n+1) scalar values instead of mn.

The image is resized to a longest edge of 128, 160 or 192 pixels without upscaling. Transparency is composited onto white. SVD is computed separately for RGB using numpy.linalg.svd; this app does not implement the decomposition from scratch.

NPZ stores U and V^T as float16 and singular values as float32, which adds quantization error. RGB intensity changes affect only PNG output. Scalar counts do not equal actual file sizes; NPZ or PNG is not guaranteed to be smaller than the input JPEG. Discarded components cannot be recovered.

## Team

Team name: LA team

| Member | SRN |
| --- | --- |
| Mohammad Aakhil Nawaz | PES1UG25AM212 |
| Madhavan Rengasamy | PES1UG25AM198 |
| Likith S Reddy | PES1UG25AM193 |
| Mohammed Zainuddin | PES1UG25AM221 |

## Files

- svd_lab.py: Streamlit app, decomposition, reconstruction and downloads.
- requirements.txt: Python dependencies.

Based on the saved November 1 SVD Image Lab application, with a zero-spectrum guard for black images.
