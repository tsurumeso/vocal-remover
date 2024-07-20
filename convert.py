import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch

from lib import dataset
from lib import nets
from lib import spec_utils

import inference


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default='models/baseline.pth')
    p.add_argument('--dataset', '-d', required=True)
    p.add_argument('--split_mode', '-S', type=str, choices=['random', 'subdirs'], default='random')
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=256)
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    model = nets.CascadedNet(args.n_fft, args.hop_length)
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

    for mix_path, inst_path in filelist:
        X_basename = os.path.splitext(os.path.basename(mix_path))[0]
        y_basename = os.path.splitext(os.path.basename(inst_path))[0]
        v_basename = X_basename + '_Vocals'
        # p_basename = X_basename + '_PseudoInstruments'

        print('converting {}...'.format(X_basename))

        X_cache_dir = os.path.join(os.path.dirname(mix_path), cache_dir)
        y_cache_dir = os.path.join(os.path.dirname(inst_path), cache_dir)
        v_dir = os.path.join(os.path.split(os.path.dirname(inst_path))[0], 'vocals')
        v_cache_dir = os.path.join(v_dir, cache_dir)

        os.makedirs(X_cache_dir, exist_ok=True)
        os.makedirs(y_cache_dir, exist_ok=True)
        os.makedirs(v_cache_dir, exist_ok=True)

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

        sp = inference.Separator(model, device, args.batchsize, args.cropsize)
        _, v = sp.separate_tta(X - y)
        # a, v = sp.separate_tta(X - y)

        # p = y + a

        # wave = spec_utils.spectrogram_to_wave(p, hop_length=args.hop_length)
        # sf.write('{}/{}.wav'.format(v_dir, p_basename), [0], sr)
        # wave = spec_utils.spectrogram_to_wave(v, hop_length=args.hop_length)
        sf.write('{}/{}.wav'.format(v_dir, v_basename), [0], sr)

        np.save('{}/{}.npy'.format(X_cache_dir, X_basename), X.transpose(2, 0, 1))
        np.save('{}/{}.npy'.format(y_cache_dir, y_basename), y.transpose(2, 0, 1))
        # np.save('{}/{}.npy'.format(y_cache_dir, p_basename), p.transpose(2, 0, 1))
        np.save('{}/{}.npy'.format(v_cache_dir, v_basename), v.transpose(2, 0, 1))


if __name__ == '__main__':
    main()
