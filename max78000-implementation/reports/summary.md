# CIFAR-10 on MCU — Phase 1 + 2 results

| Tag | Model | Params | Wt. KiB (int8) | MACs (M) | Ops paper conv. (M) | fp32 acc | int8 acc | Δ acc | Train time (s) |
|-----|-------|--------|----------------|----------|----------------------|----------|----------|-------|----------------|
| baseline_fp32 | baseline | 38,890 | 38.0 | 4.43 | 8.87 | 79.80% | 79.80% | +0.00pp | 0 |
| deeper_fp32 | deeper | 141,034 | 137.7 | 7.96 | 15.93 | 83.17% | 83.17% | +0.00pp | 0 |
| improved_fp32 | improved | 39,018 | 38.1 | 4.43 | 8.87 | 81.70% | 81.70% | +0.00pp | 0 |
| mininet_fp32 | mininet | 418,026 | 408.0 | 23.00 | 46.01 | 88.42% | 88.42% | +0.00pp | 0 |
| wide_improved_fp32 | wide_improved | 79,258 | 77.4 | 9.31 | 18.61 | 79.36% | 79.36% | +0.00pp | 0 |

**Reference (Lai et al. 2018):** 24.7 MOps/inference, 87 KB int8 weights, 79.9% int8 accuracy on CIFAR-10.

## Per-experiment notes
### baseline_fp32
- Model: `baseline` [fp32+ptq], optimizer: `adam`, lr=0.001, wd=0.0, scheduler=constant, augment=True, epochs=?
- fp32 test acc: **79.80%**, int8 test acc: **79.80%**

### deeper_fp32
- Model: `deeper` [fp32+ptq], optimizer: `adam`, lr=0.001, wd=0.0, scheduler=constant, augment=True, epochs=?
- fp32 test acc: **83.17%**, int8 test acc: **83.17%**

### improved_fp32
- Model: `improved` [fp32+ptq], optimizer: `adam`, lr=0.001, wd=0.0, scheduler=constant, augment=True, epochs=?
- fp32 test acc: **81.70%**, int8 test acc: **81.70%**

### mininet_fp32
- Model: `mininet` [fp32+ptq], optimizer: `adam`, lr=0.005, wd=0.0001, scheduler=cosine, augment=True, epochs=80
- fp32 test acc: **88.42%**, int8 test acc: **88.42%**

### wide_improved_fp32
- Model: `wide_improved` [fp32+ptq], optimizer: `adam`, lr=0.001, wd=0.0, scheduler=constant, augment=True, epochs=?
- fp32 test acc: **79.36%**, int8 test acc: **79.36%**
