# Environment Setup

This guide describes a reproducible setup for MCM. The commands assume Ubuntu/Linux, Conda, CUDA-capable NVIDIA GPU, and a fresh clone of this repository.

## Clone Repository

```bash
git clone --recursive https://github.com/hjpark83/Consistent-Scene-Understanding-in-3DGS-via-MCM.git MCM
cd MCM
```

If the repository does not include local CUDA extension folders, fetch the original 3DGS submodules manually:

```bash
mkdir -p submodules

git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git 
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git

pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

The optional feature rasterizer is only needed for `render/features.py`. If this repository already contains `submodules/diff-gaussian-rasterization-feature`, skip this step. Otherwise, place your feature-enabled rasterizer fork at:

```text
submodules/diff-gaussian-rasterization-feature
```

## Environment

Python 3.9 is recommended because it is compatible with common 3DGS/GaussianGrouping CUDA extensions.

```bash
conda create -n mcm python=3.9 -y
conda activate mcm

python -m pip install --upgrade pip setuptools wheel
```

Choose the PyTorch build that matches your GPU and driver. Install PyTorch before `requirements.txt`.

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` installs Segment Anything from GitHub and the required local 3DGS CUDA extensions:

If you do not use `requirements.txt`, install Segment Anything directly:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Download Model Checkpoints

The pipeline uses SAM from a checkpoint file, DINOv2 from Torch Hub, and Depth Anything V2 from Hugging Face. Local GitHub clones of DINOv2 or Depth Anything V2 are not required for normal execution, but the official source repositories are:

```bash
mkdir -p external
git clone https://github.com/facebookresearch/dinov2.git external/dinov2
git clone https://github.com/DepthAnything/Depth-Anything-V2.git external/Depth-Anything-V2
git clone https://github.com/facebookresearch/segment-anything.git external/segment-anything
```

### Segment Anything

Download SAM ViT-H and place it in the repository root:

```bash
wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### DINOv2

DINOv2 is loaded through `torch.hub` using `facebookresearch/dinov2`. The default model is `dinov2_vits14`, with fallback support for `dinov2_vits14_reg`.

To pre-cache DINOv2:

```bash
python - <<'PY'
import torch
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
PY
```

### Depth Anything V2

Depth Anything V2 is loaded through Hugging Face Transformers:

```text
depth-anything/Depth-Anything-V2-Small-hf
```

To pre-cache it:

```bash
python - <<'PY'
from transformers import pipeline
pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
PY
```