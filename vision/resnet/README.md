# ResNet-18 (He et al., 2015) — CIFAR-10 Implementation

**Paper:** Deep Residual Learning for Image Recognition  
**Authors:** Kaiming He et al., 2015  
**Link:** https://arxiv.org/abs/1512.03385

---

## Why this paper matters

- Introduced **residual connections** to train very deep CNNs.
- Solves vanishing gradients for deep models by learning residuals `F(x) + x`.
- Backbone of many modern architectures (detection, segmentation, etc.).
- A must-know architecture for ML interviews.

---

## Implementation details

### Model

- Architecture: **ResNet-18** (BasicBlock)
  - 4 stages with {64, 128, 256, 512} channels
  - 2 residual blocks per stage: `[2, 2, 2, 2]`
  - Each block: 3×3 conv → BN → ReLU → 3×3 conv → BN + skip connection
  - Strided convolutions at the start of stages 2–4 (downsampling)
  - Global average pooling → Linear layer to 10 classes

- Adapted for **CIFAR-10**:
  - Input: `3×32×32`
  - First conv: `3×3, stride=1, padding=1` (no 7×7 or maxpool).

### Training setup

- Dataset: **CIFAR-10** (50k train, 10k test)
- Split: 90% train / 10% validation from training set
- Input: RGB `3×32×32`
- Transforms:
  - Train: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
  - Val/Test: ToTensor + Normalize
- Loss: `CrossEntropyLoss`
- Optimizer: **SGD with momentum**
  - learning rate: `0.1`
  - momentum: `0.9`
  - weight decay: `5e-4`
- Batch size: `128`
- Epochs: `20` (adjust as needed)
- Device: `"cpu"` by default (change to `"cuda"` in config if you have GPU)

Config file: `configs/default.yaml`.

---

## How to run

From repo root:

```bash
cd vision/resnet

# create and activate venv (Windows)
python -m venv .venv
.\.venv\Scripts\activate

# install shared dependencies
pip install -r ../../requirements.txt

## Results

Training config (baseline):

- Model: ResNet-18 (CIFAR-10 variant)
- Optimizer: SGD with momentum (`lr=0.1`, `momentum=0.9`)
- Weight decay: `5e-4`
- Batch size: `128`
- Epochs: `20`
- Device: `cuda` (RTX 3060)

Final run:

```text
[epoch 20] train_acc=0.867  val_acc=0.822  best_val=0.824

## Training Curves

### Loss
![Loss](outputs/loss.png)

### Accuracy
![Accuracy](outputs/acc.png)

