<div align="center">
<img src="https://drive.google.com/uc?export=view&id=1hqIEo_SsjTLFgoA2Mv8ODDrdEUkF3awv" alt="logo" height="150"></img>
</div>
<br>

This repository is the official implementation of "Deconver: A Deconvolutional Network for Medical Image Segmentation".

--- 

**Deconver** is a segmentation architecture inspired by deconvolution techniques. We introduce a novel *mixer module* based on *nonnegative deconvolution (NDC)*, which effectively restores high-frequency details while suppressing artifacts. Built on a U-shaped structure, Deconver replaces computationally expensive self-attention blocks with this mixer to efficiently capture special dependencies.

![Graphical Abstract](https://drive.google.com/uc?export=view&id=1WhNf_Sbbe5BtfA2c1MMUU2-5KxPRYTAt)


## ✨ Key Features

- 🏆 **State-of-the-art performance** on various 2D/3D medical segmentation tasks (ISLES'22, BraTS'23, GlaS, FIVES)
- ⚡ **Significantly fewer FLOPs** compared to CNN and Transformer baselines
- 🧠 **Parameter-efficient mixer module** using a *multiplicative update rule* for the source in deconvolution


## 🔥 Latest News

- **[03.26.2024]** We have released the code for Deconver!


## 🛠️ Installation

Install Deconver directly from GitHub using:

```bash
pip install git+https://github.com/pashtari/deconver.git
```


## 🚀 Quick Start

### 2D Segmentation: GlaS Example

```python
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

x = torch.rand((1, 3, 256, 256))
y = model(x) 
```

### 3D Segmentation: ISLES'22 Example

```python
deconver = Deconver(
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
y = model(x)    # output logits
```


## 📊 Results

| Dataset       | DSC (%) ↑ | HD95 ↓ | Params (M) | FLOPs/pixel (K) |
|---------------|-----------|--------|------------|-----------------|
| ISLES'22      | 78.16     | 4.99   | 10.5       | 607.0           |
| BraTS'23      | 90.66     | 4.45   | 10.6       | 167.5           |
| GlaS          | 92.12     | 60.49  | 20.8       | 422.8           |
| FIVES         | 92.72     | 30.26  | 20.8       | 422.8           |


## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


## Contact

For questions or collaboration, contact:  
- **Pooya Ashtari**: [pooya.ash@gmail.com](mailto:pooya.ash@gmail.com)  
- GitHub: [@pashtari](https://github.com/pashtari)  