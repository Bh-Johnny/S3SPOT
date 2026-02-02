# S³POT: Contrast-Driven Face Occlusion Segmentation via Self-Supervised Prompt Learning

<div align="center">

[![Under Construction](https://img.shields.io/badge/Status-Under%20Construction-orange)](https://github.com/yourusername/yourrepo)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](<>)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Lingsong Wang¹², Mancheng Meng², Ziyan Wu², Terrence Chen², Fan Yang²\*, Dinggang Shen¹²**

¹ShanghaiTech University, ²United Imaging Intelligence

</div>

## 📖 Abstract

Existing face parsing methods usually misclassify occlusions as facial components. To deal with this, we present **S³POT**, a contrast-driven framework synergizing face generation with self-supervised spatial prompting. 

The framework consists of three modules: 
1.  **Reference Generation (RG):** Produces a clean reference image and a face parsing mask.
2.  **Feature Enhancement (FE):** Contrasts tokens between raw and reference images to obtain initial prompts.
3.  **Prompt Selection (PS):** Constructs positive/negative prompts using a greedy strategy and learns a mask decoder without occlusion ground truth.

## 🔨 Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/<YourUsername>/<YourRepoName>.git
    cd <YourRepoName>
    ```

2. **Create a virtual environment**
    ```bash
    conda create -n s3pot python=3.9
    conda activate s3pot
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You also need to install [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything).*

## 🚀 Usage

### 1. Data Preparation & Reference Generation
Before training or inference, you need to generate the **Reference Face** and the **Face Parsing Mask** (12-class segmentation of facial components) for your occluded images.

We use a GAN-based inversion method and a face parser to generate these files.

```bash
python generate_reference.py \
  --input_dir ./dataset/raw_images \
  --output_dir ./dataset/processed_data \
  --faceParser_name bisenet  # Example
```
this step will generate:

**_ref.png**: The clean reference face.

**_parsing_mask.npy**: The 12-class face parsing mask (used to guide the greedy strategy).

### 2. Training
We support two training modes as described in the paper.

**A. General Training**

Train a general model on a dataset of paired images (Occluded + Reference).

```bash
python train_general.py \
  --datapath ./dataset/processed_data \
  --bsz 3 \
  --lr 1e-4 \
  --epochs 50 \
  --output_dir ./checkpoint/train_general
```

**B. Per-Image Overfitting**

For challenging cases, you can fine-tune the model on a single image pair to achieve optimal results.

```bash
python train_per_image.py \
  --target_img ./dataset/processed_data/test_01.png \
  --ref_img ./dataset/processed_data/test_01_ref.png \
  --parsing_mask ./dataset/processed_data/test_01_parsing_mask.npy \
  --epochs 100
```
### 3. Inference
Run inference using the trained adapter. This process uses the Greedy Matching Algorithm to automatically select prompts based on the contrast between the target and reference features.

```bash
python inference.py \
  --datapath ./dataset/processed_data/val \
  --checkpoint ./checkpoint/train_general/model_epoch_50.pth \
  --output_dir ./results \
  --save_vis  # Optional: Save visualization overlay
```

## 🙏 Acknowledgements
This project is built upon Segment Anything (SAM) and E4S. We thank the authors for their great work.
