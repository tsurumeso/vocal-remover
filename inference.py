import argparse
import os

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from lib import dataset
from lib import nets
from lib import spec_utils


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--model', '-m', type=str, default='models/baseline.pth')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-l', type=int, default=1024)
    p.add_argument('--window_size', '-w', type=int, default=512)
    p.add_argument('--out_mask', '-M', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    model = nets.CascadedASPPNet()
    model.load_state_dict(torch.load(args.model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)
    print('done')

    print('loading wave source...', end=' ')
    X, sr = librosa.load(
        args.input, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
    print('done')

    print('stft of wave source...', end=' ')
    X = spec_utils.calc_spec(X, args.hop_length)
    X, phase = np.abs(X), np.exp(1.j * np.angle(X))
    coeff = X.max()
    X /= coeff
    print('done')

    offset = model.offset
    l, r, roi_size = dataset.make_padding(X.shape[2], args.window_size, offset)
    X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode='constant')
    X_roll = np.roll(X_pad, roi_size // 2, axis=2)

    model.eval()
    with torch.no_grad():
        masks = []
        masks_roll = []
        for i in tqdm(range(int(np.ceil(X.shape[2] / roi_size)))):
            start = i * roi_size
            X_window = torch.from_numpy(np.asarray([
                X_pad[:, :, start:start + args.window_size],
                X_roll[:, :, start:start + args.window_size]
            ])).to(device)
            pred = model.predict(X_window)
            pred = pred.detach().cpu().numpy()
            masks.append(pred[0])
            masks_roll.append(pred[1])

        mask = np.concatenate(masks, axis=2)[:, :, :X.shape[2]]
        mask_roll = np.concatenate(masks_roll, axis=2)[:, :, :X.shape[2]]
        mask = (mask + np.roll(mask_roll, -roi_size // 2, axis=2)) / 2

    if args.postprocess:
        vocal = X * (1 - mask) * coeff
        mask = spec_utils.mask_uninformative(mask, vocal)

    inst = X * mask * coeff
    vocal = X * (1 - mask) * coeff

    basename = os.path.splitext(os.path.basename(args.input))[0]

    print('inverse stft of instruments...', end=' ')
    wav = spec_utils.spec_to_wav(inst, phase, args.hop_length)
    print('done')
    sf.write('{}_Instruments.wav'.format(basename), wav.T, sr)

    print('inverse stft of vocals...', end=' ')
    wav = spec_utils.spec_to_wav(vocal, phase, args.hop_length)
    print('done')
    sf.write('{}_Vocals.wav'.format(basename), wav.T, sr)

    if args.out_mask:
        norm_mask = np.uint8((1 - mask) * 255).transpose(1, 2, 0)
        norm_mask = np.concatenate([
            np.max(norm_mask, axis=2, keepdims=True),
            norm_mask
        ], axis=2)[::-1]
        _, bin_mask = cv2.imencode('.png', norm_mask)
        with open('{}_Mask.png'.format(basename), mode='wb') as f:
            bin_mask.tofile(f)


if __name__ == '__main__':
    main()
