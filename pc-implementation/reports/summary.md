# CIFAR-10 on MCU — Phase 1 + 2 results

| Tag | Model | Params | Wt. KiB (int8) | MACs (M) | Ops paper conv. (M) | fp32 acc | int8 acc | Δ acc | Train time (s) |
|-----|-------|--------|----------------|----------|----------------------|----------|----------|-------|----------------|
| baseline_5x5_fp32 | baseline_5x5 | 89,578 | 87.5 | 12.30 | 24.60 | 79.79% | 79.76% | -0.03pp | 337 |
| baseline_5x5_qat | baseline_5x5 | 89,578 | 87.5 | 12.30 | 24.60 | 81.14% | 81.14% | +0.00pp | 265 |
| baseline_fp32 | baseline | 38,890 | 38.0 | 4.43 | 8.87 | 80.98% | 80.85% | -0.13pp | 270 |
| baseline_qat | baseline | 38,890 | 38.0 | 4.43 | 8.87 | 81.15% | 81.15% | +0.00pp | 253 |
| deeper_fp32 | deeper | 141,034 | 137.7 | 7.96 | 15.93 | 85.14% | 84.76% | -0.38pp | 907 |
| deeper_qat | deeper | 141,034 | 137.7 | 7.96 | 15.93 | 85.69% | 85.69% | +0.00pp | 523 |
| improved_fp32 | improved | 39,018 | 38.1 | 4.43 | 8.87 | 81.40% | 81.51% | +0.11pp | 281 |
| improved_qat | improved | 39,018 | 38.1 | 4.43 | 8.87 | 81.91% | 81.91% | +0.00pp | 268 |
| mininet_fp32 | mininet | 316,458 | 309.0 | 24.48 | 48.96 | 88.63% | 88.57% | -0.06pp | 557 |
| mininet_qat | mininet | 316,458 | 309.0 | 24.48 | 48.96 | 88.59% | 88.59% | +0.00pp | 529 |
| nascifarnet_fp32 | nascifarnet | 301,770 | 294.7 | 36.18 | 72.36 | 87.61% | 87.44% | -0.17pp | 960 |
| nascifarnet_qat | nascifarnet | 301,770 | 294.7 | 36.18 | 72.36 | 88.95% | 88.95% | +0.00pp | 635 |
| ressimplenet_fp32 | ressimplenet | 372,512 | 363.8 | 18.45 | 36.90 | 86.90% | 86.71% | -0.19pp | 1163 |
| ressimplenet_qat | ressimplenet | 372,512 | 363.8 | 18.45 | 36.90 | 88.20% | 88.20% | +0.00pp | 947 |

**Reference (Lai et al. 2018):** 24.7 MOps/inference, 87 KB int8 weights, 79.9% int8 accuracy on CIFAR-10.

## Per-experiment notes
### baseline_5x5_fp32
- Model: `baseline_5x5` [fp32+ptq], optimizer: `adam`, lr=0.001, wd=0.0, scheduler=none, augment=True, epochs=80
- fp32 test acc: **79.79%**, int8 test acc: **79.76%**

### baseline_5x5_qat
- [qat] lr=0.0005, epochs=40, augment=True, QAT switch @ epoch 5
- fp32 test acc: **81.14%**, QAT int8 test acc: **81.14%**

### baseline_fp32
- Model: `baseline` [fp32+ptq], optimizer: `adam`, lr=0.001, wd=0.0, scheduler=none, augment=True, epochs=80
- fp32 test acc: **80.98%**, int8 test acc: **80.85%**

### baseline_qat
- [qat] lr=0.0005, epochs=40, augment=True, QAT switch @ epoch 5
- fp32 test acc: **81.15%**, QAT int8 test acc: **81.15%**

### deeper_fp32
- Model: `deeper` [fp32+ptq], optimizer: `adam`, lr=0.001, wd=0.0, scheduler=none, augment=True, epochs=80
- fp32 test acc: **85.14%**, int8 test acc: **84.76%**

### deeper_qat
- [qat] lr=0.0005, epochs=40, augment=True, QAT switch @ epoch 5
- fp32 test acc: **85.69%**, QAT int8 test acc: **85.69%**

### improved_fp32
- Model: `improved` [fp32+ptq], optimizer: `adam`, lr=0.001, wd=0.0, scheduler=none, augment=True, epochs=80
- fp32 test acc: **81.40%**, int8 test acc: **81.51%**

### improved_qat
- [qat] lr=0.0005, epochs=40, augment=True, QAT switch @ epoch 5
- fp32 test acc: **81.91%**, QAT int8 test acc: **81.91%**

### mininet_fp32
- Model: `mininet` [fp32+ptq], optimizer: `adam`, lr=0.005, wd=0.0001, scheduler=cosine, augment=True, epochs=80
- fp32 test acc: **88.63%**, int8 test acc: **88.57%**

### mininet_qat
- [qat] lr=0.0001, epochs=40, augment=True, QAT switch @ epoch 5
- fp32 test acc: **88.59%**, QAT int8 test acc: **88.59%**

### nascifarnet_fp32
- Model: `nascifarnet` [fp32+ptq], optimizer: `adam`, lr=0.001, wd=0.0, scheduler=none, augment=True, epochs=80
- fp32 test acc: **87.61%**, int8 test acc: **87.44%**

### nascifarnet_qat
- [qat] lr=0.0005, epochs=40, augment=True, QAT switch @ epoch 5
- fp32 test acc: **88.95%**, QAT int8 test acc: **88.95%**

### ressimplenet_fp32
- Model: `ressimplenet` [fp32+ptq], optimizer: `adam`, lr=0.001, wd=0.0, scheduler=none, augment=True, epochs=80
- fp32 test acc: **86.90%**, int8 test acc: **86.71%**

### ressimplenet_qat
- [qat] lr=0.0005, epochs=40, augment=True, QAT switch @ epoch 5
- fp32 test acc: **88.20%**, QAT int8 test acc: **88.20%**
