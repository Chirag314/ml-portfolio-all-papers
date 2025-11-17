# Elite ML Portfolio — Paper Implementations

_Date scaffolded: 2025-11-17_

This repository is a **hands-on portfolio** of classic + modern ML papers,
with clean, reproducible subprojects. Each paper has:
- `README.md` (why it matters, key ideas, implementation plan)
- `configs/default.yaml`
- `src/train.py` & `src/eval.py` (stubs wired with unified logging)
- `notebooks/`, `data/`, `outputs/`

Generators:
```bash
make new-paper PATH=vision/resnet TITLE="Deep Residual Learning (2015)" CITATION="He et al., 2015" URL="https://arxiv.org/abs/1512.03385" DATASET="CIFAR-10" COMP="Residual Blocks" LOSS="Cross-entropy" METRICS="Top-1 Acc"

make new-exp PATH=experiments/my_exp NAME="My Exp" BRIEF="Quick idea"
```
