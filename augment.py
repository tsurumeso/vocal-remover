import argparse
import os
import subprocess

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from lib import dataset
from lib import spec_utils


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-l', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--pitch', '-p', type=int, default=-1)
    p.add_argument('--mixtures', '-m', required=True)
    p.add_argument('--instruments', '-i', required=True)
    args = p.parse_args()

    input_i = 'input_i_{}.wav'.format(args.pitch)
    input_v = 'input_v_{}.wav'.format(args.pitch)
    output_i = 'output_i_{}.wav'.format(args.pitch)
    output_v = 'output_v_{}.wav'.format(args.pitch)
    cmd_i = 'soundstretch {} {} -pitch={}'.format(input_i, output_i, args.pitch)
    cmd_v = 'soundstretch {} {} -pitch={}'.format(input_v, output_v, args.pitch)
    cache_suffix = '_pitch{}.npy'.format(args.pitch)

    cache_dir = 'sr{}_hl{}_nf{}'.format(args.sr, args. hop_length, args.n_fft)
    mix_cache_dir = os.path.join(args.mixtures, cache_dir)
    inst_cache_dir = os.path.join(args.instruments, cache_dir)
    os.makedirs(mix_cache_dir, exist_ok=True)
    os.makedirs(inst_cache_dir, exist_ok=True)

    filelist = dataset.make_pair(args.mixtures, args.instruments)
    for mix_path, inst_path in tqdm(filelist):
        mix_basename = os.path.splitext(os.path.basename(mix_path))[0]
        mix_cache_path = os.path.join(mix_cache_dir, mix_basename + cache_suffix)

        inst_basename = os.path.splitext(os.path.basename(inst_path))[0]
        inst_cache_path = os.path.join(inst_cache_dir, inst_basename + cache_suffix)

        if os.path.exists(mix_cache_path) and os.path.exists(inst_cache_path):
            continue

        X, _ = librosa.load(
            mix_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
        y, _ = librosa.load(
            inst_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')

        X, y = spec_utils.align_wave_head_and_tail(X, y, args.sr)
        v = X - y

        sf.write(input_i, y.T, args.sr)
        sf.write(input_v, v.T, args.sr)
        subprocess.call(cmd_i, stderr=subprocess.DEVNULL)
        subprocess.call(cmd_v, stderr=subprocess.DEVNULL)

        y, _ = librosa.load(
            output_i, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
        v, _ = librosa.load(
            output_v, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')

        X = y + v

        spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
        np.save(mix_cache_path, spec)

        spec = spec_utils.wave_to_spectrogram(y, args.hop_length, args.n_fft)
        np.save(inst_cache_path, spec)

        os.remove(input_i)
        os.remove(input_v)
        os.remove(output_i)
        os.remove(output_v)
