# CIFAR-10 on MCU — IMX500 results

## Compact comparison

| Tag | Model | Params | Wt. KiB (int8) | MACs (M) | Ops paper conv. (M) | fp32 acc | int8 acc | Δ acc | HW inference (ms) | Frame loop (ms) |
|-----|-------|--------|----------------|----------|----------------------|----------|----------|-------|-------------------|-----------------|
| baseline_5x5_fp32 | baseline_5x5 | 89,578 | 87.5 | 12.30 | 24.60 | 79.79% | 79.83% | +0.04pp | 1.52 | 89.39 |
| baseline_fp32 | baseline | 38,890 | 38.0 | 4.43 | 8.87 | 80.98% | 80.73% | -0.25pp | 1.53 | 87.83 |
| deeper_fp32 | deeper | 141,034 | 137.7 | 7.96 | 15.93 | 85.14% | 85.00% | -0.14pp | 1.53 | 88.89 |
| improved_fp32 | improved | 39,018 | 38.1 | 4.43 | 8.87 | 81.40% | 81.48% | +0.08pp | 1.53 | 86.40 |
| mininet_fp32 | mininet | 316,458 | 309.0 | 24.48 | 48.96 | 88.63% | 88.48% | -0.15pp | 1.53 | 92.28 |

## Full sweep

| Tag | Model | Params | Wt. KiB (int8) | MACs (M) | Ops paper conv. (M) | fp32 acc | int8 acc | Δ acc | HW inference (ms) | Frame loop (ms) |
|-----|-------|--------|----------------|----------|----------------------|----------|----------|-------|-------------------|-----------------|
| baseline_5x5_fp32 | baseline_5x5 | 89,578 | 87.5 | 12.30 | 24.60 | 79.79% | 79.83% | +0.04pp | 1.52 | 89.39 |
| baseline_5x5_fp32_ptq | baseline_5x5 | 89,578 | 87.5 | 12.30 | 24.60 | 79.85% | 79.77% | -0.08pp | 1.53 | 89.66 |
| baseline_5x5_qat | baseline_5x5 | 89,578 | 87.5 | 12.30 | 24.60 | 81.14% | 81.04% | -0.10pp | 1.52 | 89.88 |
| baseline_fp32 | baseline | 38,890 | 38.0 | 4.43 | 8.87 | 80.98% | 80.73% | -0.25pp | 1.53 | 87.83 |
| baseline_fp32_ptq | baseline | 38,890 | 38.0 | 4.43 | 8.87 | 80.91% | 80.77% | -0.14pp | 1.53 | 87.54 |
| baseline_qat | baseline | 38,890 | 38.0 | 4.43 | 8.87 | 81.15% | 80.94% | -0.21pp | 1.53 | 86.45 |
| deeper_fp32 | deeper | 141,034 | 137.7 | 7.96 | 15.93 | 85.14% | 85.00% | -0.14pp | 1.53 | 88.89 |
| deeper_fp32_ptq | deeper | 141,034 | 137.7 | 7.96 | 15.93 | 21.47% | 21.18% | -0.29pp | 1.53 | 88.84 |
| deeper_qat | deeper | 141,034 | 137.7 | 7.96 | 15.93 | 85.69% | 85.62% | -0.07pp | 1.53 | 89.55 |
| improved_fp32 | improved | 39,018 | 38.1 | 4.43 | 8.87 | 81.40% | 81.48% | +0.08pp | 1.53 | 86.40 |
| improved_fp32_ptq | improved | 39,018 | 38.1 | 4.43 | 8.87 | 39.23% | 39.04% | -0.19pp | 1.52 | 85.77 |
| improved_qat | improved | 39,018 | 38.1 | 4.43 | 8.87 | 81.91% | 81.75% | -0.16pp | 1.53 | 87.31 |
| mininet_fp32 | mininet | 316,458 | 309.0 | 24.48 | 48.96 | 88.63% | 88.48% | -0.15pp | 1.53 | 92.28 |
| mininet_fp32_ptq | mininet | 316,458 | 309.0 | 24.48 | 48.96 | 13.01% | 12.79% | -0.22pp | 1.52 | 91.45 |
| mininet_qat | mininet | 316,458 | 309.0 | 24.48 | 48.96 | 88.59% | 88.52% | -0.07pp | 1.53 | 91.18 |
| nascifarnet_fp32 | nascifarnet | 301,770 | 294.7 | 36.18 | 72.36 | 87.61% | 87.64% | +0.03pp | 1.53 | 90.63 |
| nascifarnet_fp32_ptq | nascifarnet | 301,770 | 294.7 | 36.18 | 72.36 | 10.10% | 10.09% | -0.01pp | 1.53 | 90.87 |
| nascifarnet_qat | nascifarnet | 301,770 | 294.7 | 36.18 | 72.36 | 88.95% | 88.95% | +0.00pp | 1.53 | 90.34 |
| ressimplenet_fp32 | ressimplenet | 372,512 | 363.8 | 18.45 | 36.90 | 86.90% | 86.97% | +0.07pp | 1.52 | 92.29 |
| ressimplenet_fp32_ptq | ressimplenet | 372,512 | 363.8 | 18.45 | 36.90 | 9.99% | 10.00% | +0.01pp | 1.53 | 91.11 |
| ressimplenet_qat | ressimplenet | 372,512 | 363.8 | 18.45 | 36.90 | 88.20% | 88.32% | +0.12pp | 1.53 | 90.84 |

**Reference (Lai et al. 2018):** 24.7 MOps/inference, 87 KB int8 weights, 79.9% int8 accuracy on CIFAR-10.

## Per-experiment notes
### baseline_5x5_fp32
- Model: `baseline_5x5`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.52 ms mean, 1.52 ms p50, 1.98 ms p95**
- fp32 test acc: **79.79%**, int8 test acc: **79.83%**

### baseline_5x5_fp32_ptq
- Model: `baseline_5x5`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.52 ms p50, 1.98 ms p95**
- fp32 test acc: **79.85%**, int8 test acc: **79.77%**

### baseline_5x5_qat
- Model: `baseline_5x5`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.52 ms mean, 1.52 ms p50, 1.97 ms p95**
- fp32 test acc: **81.14%**, int8 test acc: **81.04%**

### baseline_fp32
- Model: `baseline`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.53 ms p50, 1.98 ms p95**
- fp32 test acc: **80.98%**, int8 test acc: **80.73%**

### baseline_fp32_ptq
- Model: `baseline`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.53 ms p50, 1.97 ms p95**
- fp32 test acc: **80.91%**, int8 test acc: **80.77%**

### baseline_qat
- Model: `baseline`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.52 ms p50, 1.98 ms p95**
- fp32 test acc: **81.15%**, int8 test acc: **80.94%**

### deeper_fp32
- Model: `deeper`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.53 ms p50, 1.98 ms p95**
- fp32 test acc: **85.14%**, int8 test acc: **85.00%**

### deeper_fp32_ptq
- Model: `deeper`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.53 ms p50, 1.98 ms p95**
- fp32 test acc: **21.47%**, int8 test acc: **21.18%**

### deeper_qat
- Model: `deeper`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.53 ms p50, 1.97 ms p95**
- fp32 test acc: **85.69%**, int8 test acc: **85.62%**

### improved_fp32
- Model: `improved`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.53 ms p50, 1.98 ms p95**
- fp32 test acc: **81.40%**, int8 test acc: **81.48%**

### improved_fp32_ptq
- Model: `improved`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.52 ms mean, 1.52 ms p50, 1.98 ms p95**
- fp32 test acc: **39.23%**, int8 test acc: **39.04%**

### improved_qat
- Model: `improved`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.53 ms p50, 1.97 ms p95**
- fp32 test acc: **81.91%**, int8 test acc: **81.75%**

### mininet_fp32
- Model: `mininet`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.53 ms p50, 1.98 ms p95**
- fp32 test acc: **88.63%**, int8 test acc: **88.48%**

### mininet_fp32_ptq
- Model: `mininet`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.52 ms mean, 1.52 ms p50, 1.97 ms p95**
- fp32 test acc: **13.01%**, int8 test acc: **12.79%**

### mininet_qat
- Model: `mininet`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.53 ms p50, 1.97 ms p95**
- fp32 test acc: **88.59%**, int8 test acc: **88.52%**

### nascifarnet_fp32
- Model: `nascifarnet`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.53 ms p50, 1.97 ms p95**
- fp32 test acc: **87.61%**, int8 test acc: **87.64%**

### nascifarnet_fp32_ptq
- Model: `nascifarnet`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.53 ms p50, 1.98 ms p95**
- fp32 test acc: **10.10%**, int8 test acc: **10.09%**

### nascifarnet_qat
- Model: `nascifarnet`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.52 ms p50, 1.98 ms p95**
- fp32 test acc: **88.95%**, int8 test acc: **88.95%**

### ressimplenet_fp32
- Model: `ressimplenet`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.52 ms mean, 1.52 ms p50, 1.97 ms p95**
- fp32 test acc: **86.90%**, int8 test acc: **86.97%**

### ressimplenet_fp32_ptq
- Model: `ressimplenet`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.53 ms p50, 1.97 ms p95**
- fp32 test acc: **9.99%**, int8 test acc: **10.00%**

### ressimplenet_qat
- Model: `ressimplenet`, optimizer: `-`, lr=?, batch_size=?, epochs=?, augment=False
- HW inference time: **1.53 ms mean, 1.53 ms p50, 1.98 ms p95**
- fp32 test acc: **88.20%**, int8 test acc: **88.32%**
## Converter memory checks

| Tag | Fit in chip | Runtime KB | Model KB | Usage | Utilization |
|-----|-------------|------------|----------|-------|-------------|
| baseline_5x5_fp32_imx500_ptq | True | 47.00KB | 144.06KB | 192.06KB | 3% |
| baseline_5x5_fp32_ptq_imx500_ptq | True | 47.00KB | 144.06KB | 192.06KB | 3% |
| baseline_5x5_qat_imx500_ptq | True | 47.00KB | 144.06KB | 192.06KB | 3% |
| baseline_fp32_imx500_ptq | True | 47.00KB | 69.06KB | 117.06KB | 2% |
| baseline_fp32_ptq_imx500_ptq | True | 47.00KB | 69.06KB | 117.06KB | 2% |
| baseline_qat_imx500_ptq | True | 47.00KB | 69.06KB | 117.06KB | 2% |
| deeper_fp32_imx500_ptq | True | 47.00KB | 150.16KB | 198.16KB | 3% |
| deeper_fp32_ptq_imx500_ptq | True | 47.00KB | 150.16KB | 198.16KB | 3% |
| deeper_qat_imx500_ptq | True | 47.00KB | 150.16KB | 198.16KB | 3% |
| improved_fp32_imx500_ptq | True | 47.00KB | 69.06KB | 117.06KB | 2% |
| improved_fp32_ptq_imx500_ptq | True | 47.00KB | 69.06KB | 117.06KB | 2% |
| improved_qat_imx500_ptq | True | 47.00KB | 69.06KB | 117.06KB | 2% |
| mininet_fp32_imx500_ptq | True | 47.00KB | 322.09KB | 370.09KB | 5% |
| mininet_fp32_ptq_imx500_ptq | True | 47.00KB | 322.09KB | 370.09KB | 5% |
| mininet_qat_imx500_ptq | True | 47.00KB | 322.09KB | 370.09KB | 5% |
| nascifarnet_fp32_imx500_ptq | True | 103.00KB | 327.19KB | 431.19KB | 6% |
| nascifarnet_fp32_ptq_imx500_ptq | True | 103.00KB | 327.19KB | 431.19KB | 6% |
| nascifarnet_qat_imx500_ptq | True | 103.00KB | 327.19KB | 431.19KB | 6% |
| ressimplenet_fp32_imx500_ptq | True | 94.00KB | 465.69KB | 560.69KB | 7% |
| ressimplenet_fp32_ptq_imx500_ptq | True | 94.00KB | 465.69KB | 560.69KB | 7% |
| ressimplenet_qat_imx500_ptq | True | 94.00KB | 465.69KB | 560.69KB | 7% |

## Live Pi logs

| Tag | Samples | Frame limit | HW mean ms | HW p50 ms | HW p95 ms | Frame mean ms | Frame p50 ms | Frame p95 ms |
|-----|---------|-------------|------------|-----------|-----------|--------------|-------------|-------------|
| baseline_5x5_fp32_imx500_ptq | 503 | 500 | 1.52 | 1.52 | 1.98 | 89.39 | 83.39 | 86.61 |
| baseline_5x5_fp32_ptq_imx500_ptq | 501 | 500 | 1.53 | 1.52 | 1.98 | 89.66 | 83.37 | 86.54 |
| baseline_5x5_qat_imx500_ptq | 500 | 500 | 1.52 | 1.52 | 1.97 | 89.88 | 83.41 | 86.78 |
| baseline_fp32_imx500_ptq | 501 | 500 | 1.53 | 1.53 | 1.98 | 87.83 | 83.38 | 86.70 |
| baseline_fp32_ptq_imx500_ptq | 501 | 500 | 1.53 | 1.53 | 1.97 | 87.54 | 83.40 | 86.39 |
| baseline_qat_imx500_ptq | 502 | 500 | 1.53 | 1.52 | 1.98 | 86.45 | 83.38 | 86.75 |
| deeper_fp32_imx500_ptq | 500 | 500 | 1.53 | 1.53 | 1.98 | 88.89 | 83.39 | 86.73 |
| deeper_fp32_ptq_imx500_ptq | 505 | 500 | 1.53 | 1.53 | 1.98 | 88.84 | 83.44 | 86.26 |
| deeper_qat_imx500_ptq | 501 | 500 | 1.53 | 1.53 | 1.97 | 89.55 | 83.38 | 85.97 |
| improved_fp32_imx500_ptq | 502 | 500 | 1.53 | 1.53 | 1.98 | 86.40 | 83.38 | 85.79 |
| improved_fp32_ptq_imx500_ptq | 501 | 500 | 1.52 | 1.52 | 1.98 | 85.77 | 83.38 | 86.49 |
| improved_qat_imx500_ptq | 503 | 500 | 1.53 | 1.53 | 1.97 | 87.31 | 83.40 | 86.63 |
| mininet_fp32_imx500_ptq | 503 | 500 | 1.53 | 1.53 | 1.98 | 92.28 | 83.40 | 86.75 |
| mininet_fp32_ptq_imx500_ptq | 502 | 500 | 1.52 | 1.52 | 1.97 | 91.45 | 83.40 | 85.50 |
| mininet_qat_imx500_ptq | 503 | 500 | 1.53 | 1.53 | 1.97 | 91.18 | 83.38 | 86.57 |
| nascifarnet_fp32_imx500_ptq | 501 | 500 | 1.53 | 1.53 | 1.97 | 90.63 | 83.38 | 85.54 |
| nascifarnet_fp32_ptq_imx500_ptq | 501 | 500 | 1.53 | 1.53 | 1.98 | 90.87 | 83.38 | 86.33 |
| nascifarnet_qat_imx500_ptq | 503 | 500 | 1.53 | 1.52 | 1.98 | 90.34 | 83.37 | 86.44 |
| ressimplenet_fp32_imx500_ptq | 503 | 500 | 1.52 | 1.52 | 1.97 | 92.29 | 83.38 | 86.15 |
| ressimplenet_fp32_ptq_imx500_ptq | 504 | 500 | 1.53 | 1.53 | 1.97 | 91.11 | 83.39 | 86.63 |
| ressimplenet_qat_imx500_ptq | 505 | 500 | 1.53 | 1.53 | 1.98 | 90.84 | 83.39 | 86.69 |

