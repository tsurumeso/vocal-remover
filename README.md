# vocal-remover
This is a spectrogram based vocal seperation tool using deep neural networks.

## Requirements

  - Chainer
  - CuPy (for GPU support)
  - LibROSA

## Installation

### Install Python packages
```
pip install chainer
pip install librosa
```

### Enable GPU support

Install CuPy precompiled binary package which includes the latest version of cuDNN library.
See: [CuPy Installation Guide](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy)

### Getting vocal-remover
Download the latest version from [here](https://github.com/tsurumeso/vocal-remover/releases).

## Usage

```
python inference.py --input path/to/audio/file --gpu 0
```

## References

- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference