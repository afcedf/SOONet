# Scanning Only Once: An End-to-end Framework for Fast Temporal Grounding in Long Videos

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2303.08345)

This repository is an official implementation of [SOONet](https://arxiv.org/abs/2303.08345). Code will be released soon.

## Overview

![Framework](figs/framework.png)

SOONet is an end-to-end framework for temporal grounding in long videos. It manages to model an hours-long video with one-time network execution, alleviating the inefficiency issue caused by the sliding window pipeline. Besides, it integrates both inter-anchor context knowledge and intra-anchor content knowledge with carefully tailored network structure and training objectives, leading to accurate temporal boundary localization. SOONet achieves state-of-the-art performance on MAD and Ego4d, regarding both accuracy and efficiency.

## Main Results

We provide results on MAD **test set**. More detailed experimental results can be found in the paper. Models are trained and tested with *1x A100-80G*.

### **Accuracy Comparison**

| Method  | R1@0.1 | R5@0.1 | R50@0.1 | R1@0.3 | R5@0.3 | R50@0.3 | R1@0.5 | R5@0.5 | R50@0.5 |
|:-------:|:------:|:------:|:-------:|:------:|:------:|:-------:|:------:|:------:|:-------:|
| VLG-Net | 3.64   | 11.66  | 39.78   | 2.76   | 9.31   | 34.27   | 1.65   | 5.99   | 24.93   |
| CLIP    | 6.57   | 15.05  | 37.92   | 3.13   | 9.85   | 28.71   | 1.39   | 5.44   | 18.80   |
| CONE    | 8.90   | 20.51  | 43.36   | 6.87   | 16.11  | 34.73   | 4.10   | 9.59   | 20.56   |
| **SOONet** | **11.26** | **23.21** | **50.32** | **9.00** | **19.64** | **44.78** | **5.32** | **13.14** | **32.59** |

### **Efficiency Comparison**

| Method  | Params | FLOPs | GPU Mem. | Execution Time |
|:-------:|:------:|:-----:|:--------:|:--------------:|
| CLIP    |  0     | 0.2G  | 2.9G     |  7387.8s       |
| VLG-Net | 5,330,435 | 1757.3G | 20.0G | 29556.0s     |
| **SOONet** | 22,970,947 | 70.2G | 2.4G | **505.0s**  |

## Qualitative Results

![Visualization](figs/visualization.png)

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{pan2023scanning,
  title={Scanning Only Once: An End-to-end Framework for Fast Temporal Grounding in Long Videos},
  author={Pan, Yulin and He, Xiangteng and Gong, Biao and Lv, Yiliang and Shen, Yujun and Peng, Yuxin and Zhao, Deli},
  journal={arXiv preprint arXiv:2303.08345},
  year={2023}
}
```

## Acknowledgement

Our code references the following projects. Many thanks to the authors.

* [Swin-Transformer-1D](https://github.com/meraks/Swin-Transformer-1D.git)
* [Tensorflow-Ranking](https://github.com/tensorflow/ranking.git)
