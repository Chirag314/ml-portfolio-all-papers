# LeNet-5 (LeCun et al., 1998) — MNIST Implementation

**Paper:** Gradient-Based Learning Applied to Document Recognition  
**Authors:** Yann LeCun et al., 1998  
**Link:** http://yann.lecun.com/exdb/lenet/

---

## Why this paper matters

- One of the earliest successful **convolutional neural networks** for vision.
- Established the pattern: `Conv → Nonlinearity → Pooling → Fully Connected`.
- Historically important for handwritten digit recognition (MNIST) and a great
  “hello world” for image CNNs in an ML portfolio.

---

## Implementation details

**Model**

- Architecture: LeNet-5 (simplified connectivity)

  - C1: `Conv2d(1 → 6, kernel=5, stride=1, padding=2)`
  - S2: `AvgPool2d(2×2)`  
  - C3: `Conv2d(6 → 16, kernel=5, stride=1)`
  - S4: `AvgPool2d(2×2)`
  - C5: `Conv2d(16 → 120, kernel=5, stride=1)`
  - F6: `Linear(120 → 84)`
  - Output: `Linear(84 → 10)`
  - Activation: **Tanh** after each conv / FC (as in the original paper).

- Partial connectivity in C3 from the paper is simplified to **full conv**
  (common in modern LeNet re-implementations).

**Training setup**

- Dataset: **MNIST** (train: 60k, test: 10k)
- Input: grayscale, 1×28×28
- Loss: `CrossEntropyLoss`
- Optimizer (this run): **SGD + Momentum**

  - learning rate: `0.01`
  - momentum: `0.9`
  - weight decay: `5e-4`

- Batch size: `128`
- Epochs: `5`
- Device: `cpu` (PyTorch CPU build)

Config lives in: `configs/default.yaml`.

---

## Results

Training run:

```text
[epoch 5] train_acc=0.977  eval_acc=0.980  best=0.980
