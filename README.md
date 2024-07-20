# vocal-remover

[![Release](https://img.shields.io/github/release/tsurumeso/vocal-remover.svg)](https://github.com/tsurumeso/vocal-remover/releases/latest)
[![Release](https://img.shields.io/github/downloads/tsurumeso/vocal-remover/total.svg)](https://github.com/tsurumeso/vocal-remover/releases)

This is a deep-learning-based tool to extract instrumental track from your songs.

## Installation

### Getting vocal-remover
Download the latest version from [here](https://github.com/tsurumeso/vocal-remover/releases).

### Install PyTorch
**See**: [GET STARTED](https://pytorch.org/get-started/locally/)

### Install the other packages
```
cd vocal-remover
pip install -r requirements.txt
```

## Usage
The following command separates the input into instrumental and vocal tracks. They are saved as `*_Instruments.wav` and `*_Vocals.wav`.

### Run on CPU
```
python inference.py --input path/to/an/audio/file
```

### Run on GPU
```
python inference.py --input path/to/an/audio/file --gpu 0
```

### Advanced options
`--tta` option performs Test-Time-Augmentation to improve the separation quality.
```
python inference.py --input path/to/an/audio/file --tta --gpu 0
```

`--postprocess` option masks instrumental part based on the vocals volume to improve the separation quality.
> [!WARNING]
> This is an experimental feature. If you get any problems with this option, please disable it.
```
python inference.py --input path/to/an/audio/file --postprocess --gpu 0
```

## Train your own model

### Place your dataset
```
path/to/dataset/
  +- instruments/
  |    +- 01_foo_inst.wav
  |    +- 02_bar_inst.mp3
  |    +- ...
  +- mixtures/
       +- 01_foo_mix.wav
       +- 02_bar_mix.mp3
       +- ...
```

### Train a model
```
python train.py --dataset path/to/dataset --mixup_rate 0.5 --reduction_rate 0.5 --gpu 0
```

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Choi et al., "PHASE-AWARE SPEECH ENHANCEMENT WITH DEEP COMPLEX U-NET", https://openreview.net/pdf?id=SkeRTsAcYm
- [5] Jansson et al., "Learned complex masks for multi-instrument source separation", https://arxiv.org/pdf/2103.12864.pdf
- [6] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
