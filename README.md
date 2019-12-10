# vocal-remover

[![Release](https://img.shields.io/github/release/tsurumeso/vocal-remover.svg)](https://github.com/tsurumeso/vocal-remover/releases/latest)
[![Release](https://img.shields.io/github/downloads/tsurumeso/vocal-remover/total.svg)](https://github.com/tsurumeso/vocal-remover/releases)

This is a deep-learning-based tool to extract instrumental track from mixture audio.

## Installation

### Install required packages
```
pip install -r requirements.txt
```

### Getting vocal-remover
Download the latest version from [here](https://github.com/tsurumeso/vocal-remover/releases).

## Usage
Running the following code will split the mixture audio into an instrumental track and a vocal track. They are saved as `instrumental.wav` and `vocal.wav`.

### Run on CPU
```
python inference.py --input path/to/mixture/audio
```

### Run on GPU
```
python inference.py --input path/to/mixture/audio --gpu 0
```

## Train your own model
```
python train.py -i dataset/instrumentals -m dataset/mixtures -M -g 0
```

`-i` specifies an instrumental audio directory, and `-m` specifies the corresponding mixture audio directory.

```
dataset/
  +- instrumentals/
  |    +- 01_foo_inst.wav
  |    +- 02_bar_inst.mp3
  |    +- ...
  +- mixtures/
       +- 01_foo_mix.wav
       +- 02_bar_mix.mp3
       +- ...
```

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference