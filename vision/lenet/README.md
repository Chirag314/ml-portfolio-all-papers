# LeNet-5 (LeCun et al., 1998) — MNIST

Paper: **Gradient-Based Learning Applied to Document Recognition**  
Link: <http://yann.lecun.com/exdb/lenet/>

## Why it matters

- One of the earliest successful convolutional architectures.
- Introduces the basic CNN pattern: conv → nonlinearity → pooling → fully-connected.
- Good “Hello World” for vision papers in your ML portfolio.

## Implementation plan

- Dataset: MNIST (handwritten digits)
- Model: LeNet-5 (C1–S2–C3–S4–C5–F6–Output, Tanh + AvgPool)
- Loss: Cross-entropy for 10-class classification
- Metrics: Accuracy on train / test

## How to run (from this folder)

```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scripts\activate

pip install -r ../../requirements.txt   # use global deps
python -m src.train --config configs/default.yaml
