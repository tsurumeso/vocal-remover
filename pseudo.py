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
    p.add_argument('--mixtures', '-m', required=True)
    p.add_argument('--instruments', '-i', required=True)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=256)
    p.add_argument('--postprocess', '-p', action='store_true')
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    model = nets.CascadedNet(args.n_fft, args.hop_length)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)
    print('done')

    filelist = dataset.make_pair(args.mixtures, args.instruments)
    for mix_path, inst_path in filelist:
        # if '_mixture' in mix_path and '_inst' in inst_path:
        #     continue
        # else:
        #     pass

        basename = os.path.splitext(os.path.basename(mix_path))[0]
        print(basename)

        print('loading wave source...', end=' ')
        X, sr = librosa.load(
            mix_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
        y, sr = librosa.load(
            inst_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
        print('done')

        if X.ndim == 1:
            # mono to stereo
            X = np.asarray([X, X])

        print('stft of wave source...', end=' ')
        X, y = spec_utils.align_wave_head_and_tail(X, y, sr)
        X = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
        y = spec_utils.wave_to_spectrogram(y, args.hop_length, args.n_fft)
        print('done')

        sp = inference.Separator(model, device, args.batchsize, args.cropsize, args.postprocess)
        a_spec, _ = sp.separate_tta(X - y)

        print('inverse stft of pseudo instruments...', end=' ')
        pseudo_inst = y + a_spec
        print('done')

        sf.write('pseudo/{}_PseudoInstruments.wav'.format(basename), [0], sr)
        np.save('pseudo/{}_PseudoInstruments.npy'.format(basename), pseudo_inst)


if __name__ == '__main__':
    main()
