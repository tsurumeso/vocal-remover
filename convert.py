import argparse
import os
# import re

import librosa
import numpy as np
import soundfile as sf
import torch

from lib import dataset
from lib import nets
from lib import spec_utils

import inference


MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'baseline.pth')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument('--dataset', '-d', required=True)
    p.add_argument('--split_mode', '-S', type=str, choices=['random', 'subdirs'], default='random')
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=256)
    p.add_argument('--complex', '-X', action='store_true')
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    model = nets.CascadedNet(args.n_fft, args.hop_length, is_complex=args.complex)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)
    print('done')

    cache_dir = 'sr{}_hl{}_nf{}'.format(args.sr, args.hop_length, args.n_fft)
    filelist = dataset.raw_data_split(
        dataset_dir=args.dataset,
        split_mode=args.split_mode
    )

    sp = inference.Separator(model, device, args.batchsize, args.cropsize)

    for mix_path, inst_path in filelist:
        X_basename = os.path.splitext(os.path.basename(mix_path))[0]
        y_basename = os.path.splitext(os.path.basename(inst_path))[0]
        pv_basename = X_basename + '_PseudoVocals'
        # pi_basename = X_basename + '_PseudoInstruments'

        print('converting {}...'.format(X_basename))

        X_dir = os.path.dirname(mix_path)
        y_dir = os.path.dirname(inst_path)
        pv_dir = os.path.join(os.path.split(y_dir)[0], 'pseudo_vocals')
        pi_dir = os.path.join(os.path.split(y_dir)[0], 'pseudo_instruments')

        X_cache_dir = os.path.join(X_dir, cache_dir)
        y_cache_dir = os.path.join(y_dir, cache_dir)
        pv_cache_dir = os.path.join(pv_dir, cache_dir)
        pi_cache_dir = os.path.join(pi_dir, cache_dir)

        os.makedirs(X_cache_dir, exist_ok=True)
        os.makedirs(y_cache_dir, exist_ok=True)
        os.makedirs(pv_cache_dir, exist_ok=True)
        os.makedirs(pi_cache_dir, exist_ok=True)

        X, sr = librosa.load(
            mix_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
        y, sr = librosa.load(
            inst_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')

        if X.ndim == 1:
            # mono to stereo
            X = np.asarray([X, X])

        X, y = spec_utils.align_wave_head_and_tail(X, y, sr)
        X = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
        y = spec_utils.wave_to_spectrogram(y, args.hop_length, args.n_fft)

        # if re.match(r'\d{3}_mixture', X_basename) and re.match(r'\d{3}_inst', y_basename):
        #     print('this is DSD100 Dataset')
        #     pv = X - y
        #     pi = y
        # else:
        _, pv = sp.separate_tta(X - y)
        # pa, pv = sp.separate_tta(X - y)
        # pi = y + pa

        wave = spec_utils.spectrogram_to_wave(pv, hop_length=args.hop_length)
        sf.write('{}/{}.wav'.format(pv_dir, pv_basename), wave.T, sr)
        # wave = spec_utils.spectrogram_to_wave(pi, hop_length=args.hop_length)
        # sf.write('{}/{}.wav'.format(pi_dir, pi_basename), wave.T, sr)

        np.save('{}/{}.npy'.format(X_cache_dir, X_basename), X.transpose(2, 0, 1))
        np.save('{}/{}.npy'.format(y_cache_dir, y_basename), y.transpose(2, 0, 1))
        np.save('{}/{}.npy'.format(pv_cache_dir, pv_basename), pv.transpose(2, 0, 1))
        # np.save('{}/{}.npy'.format(pi_cache_dir, pi_basename), pi.transpose(2, 0, 1))


if __name__ == '__main__':
    main()
