<div align="center">
<img src="https://www.dropbox.com/scl/fi/lzxwb33sjwi2nfmudme5z/logo.png?rlkey=lqvphiuirl7qak5d8hjyeky8j&st=k0cgmsuy&raw=1" alt="Deconver Logo" height="150"></img>
<br><br>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.00302-b31b1b.svg)](https://arxiv.org/abs/2504.00302)
</div>

This repository is the official implementation of ["Deconver: A Deconvolutional Network for Medical Image Segmentation"](https://arxiv.org/abs/2504.00302).

--- 

**Deconver** is a segmentation architecture inspired by deconvolution techniques. We introduce a novel *mixer module* based on *nonnegative deconvolution (NDC)*, which effectively restores high-frequency details while suppressing artifacts. Built on a U-shaped backbone, Deconver replaces computationally expensive self-attention blocks with this mixer to efficiently capture special dependencies.

![Graphical Abstract](https://www.dropbox.com/scl/fi/fb28d3lfmxd0ptj2ez2am/graphical_abstract.png?rlkey=hjzn5i5pw7ciksz9e15j851zq&st=fj6gc5ic&raw=1)


## ✨ Key Features

- 🏆 **State-of-the-art performance** on various 2D/3D medical segmentation tasks (ISLES'22, BraTS'23, GlaS, FIVES)
- ⚡ **Significantly fewer FLOPs** compared to CNN and Transformer baselines
- 🧠 **Parameter-efficient mixer module** using a *multiplicative update rule* for the source in deconvolution


## 🔥 Latest News

- **[April 2, 2024]** The preprint of our paper is available on [arXiv](https://arxiv.org/abs/2504.00302)!
- **[March 26, 2024]** The source code for Deconver is released!


## 🛠️ Installation

Install Deconver directly from GitHub using:

```bash
pip install git+https://github.com/pashtari/deconver.git
```


## 🚀 Quick Start

### 2D Segmentation: GlaS Example

```python
import torch
import torch.nn as nn
from deconver import Deconver

model = Deconver(
    in_channels=3,
    out_channels=1,
    spatial_dims=2,
    encoder_depth=(1, 1, 1, 1, 1),
    encoder_width=(32, 64, 128, 256, 512),
    strides=(1, 2, 2, 2, 2),
    decoder_depth=(1, 1, 1, 1),
    norm=nn.InstanceNorm2d,
    kernel_size=(5, 5),
    groups=-1,              # depth-wise grouping
    ratio=4,                # source channel expansion ratio
)

x = torch.rand(1, 3, 256, 256)
y = model(x)                # output logits
```

### 3D Segmentation: ISLES'22 Example

```python
model = Deconver(
    in_channels=2,
    out_channels=1,
    spatial_dims=3,
    encoder_depth=(1, 1, 1, 1),
    encoder_width=(64, 128, 256, 512),
    strides=(1, 2, 2, 2),
    decoder_depth=(1, 1, 1),
    norm=nn.InstanceNorm3d,
    kernel_size=(3, 3, 3),
    groups=-1,
    ratio=4, 
)

x = torch.rand(1, 2, 64, 64, 64)
y = model(x)
```


## 📊 Results

| Dataset       | DSC (%) ↑ | HD95 ↓ | Params (M) | FLOPs / pixel (K) |
|---------------|-----------|--------|------------|-----------------|
| ISLES'22      | 78.16     | 4.99   | 10.5       | 607.0           |
| BraTS'23      | 90.66     | 4.45   | 10.6       | 167.5           |
| GlaS          | 92.12     | 60.49  | 20.8       | 422.8           |
| FIVES         | 92.72     | 30.26  | 20.8       | 422.8           |

*Metrics: Dice Similarity Coefficient (DSC), 95th percentile Hausdorff Distance (HD95)*


## 📜 Citation

```bibtex
@article{ashtari2024deconver,
  title={Deconver: A Deconvolutional Network for Medical Image Segmentation},
  author={Ashtari, Pooya and others},
  journal={arXiv preprint arXiv:2504.00302},
  year={2024}
}
```


## 📄 License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


## 📬 Contact

For questions or collaboration, contact:  
- **Pooya Ashtari**: [pooya.ash@gmail.com](mailto:pooya.ash@gmail.com)  
- GitHub: [@pashtari](https://github.com/pashtari)