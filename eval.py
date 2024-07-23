import argparse
import os

import librosa
import museval
import numpy as np
import torch

from lib import nets
from lib import spec_utils

import inference


MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'baseline.pth')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=256)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--output_dir', '-o', type=str, default="")
    p.add_argument('--complex', '-X', action='store_true')
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    if args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(args.gpu))
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
    model = nets.CascadedNet(args.n_fft, args.hop_length, 32, 128, args.complex)
    model.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
    model.to(device)
    print('done')

    sp = inference.Separator(
        model=model,
        device=device,
        batchsize=args.batchsize,
        cropsize=args.cropsize
    )

    all = []
    dirs = os.listdir(args.input)
    for dir in dirs:
        print(dir, end=' ')
        dir = os.path.join(args.input, dir)
        bass, _ = librosa.load(
            os.path.join(dir, 'bass.wav'), sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_best'
        )
        drums, _ = librosa.load(
            os.path.join(dir, 'drums.wav'), sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_best'
        )
        other, _ = librosa.load(
            os.path.join(dir, 'other.wav'), sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_best'
        )
        vocals, _ = librosa.load(
            os.path.join(dir, 'vocals.wav'), sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_best'
        )
        y = bass + drums + other
        X = y + vocals
        print('done')

        print('stft of wave source...', end=' ')
        X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
        print('done')

        if args.tta:
            y_spec, v_spec = sp.separate_tta(X_spec)
        else:
            y_spec, v_spec = sp.separate(X_spec)

        y_wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
        v_wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)

        SDR, ISR, SIR, SAR = museval.evaluate(
            [y.T, vocals.T], [y_wave.T, v_wave.T]
        )

        sdr = np.nanmean(SDR, axis=1)
        isr = np.nanmean(ISR, axis=1)
        sir = np.nanmean(SIR, axis=1)
        sar = np.nanmean(SAR, axis=1)

        print(sdr)
        print(isr)
        print(sir)
        print(sar)

        all.append([sdr, isr, sir, sar])

    print(np.asarray(all).mean(axis=0))


if __name__ == '__main__':
    main()
