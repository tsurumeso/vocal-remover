import argparse

import torch
import cv2
import librosa
import numpy as np
from tqdm import tqdm

from lib import spec_utils
from lib import unet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--model', '-m', type=str, default='models/baseline.npz')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-l', type=int, default=1024)
    p.add_argument('--window_size', '-w', type=int, default=1024)
    p.add_argument('--out_mask', '-M', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    args = p.parse_args()

    print('loading model...', end=' ')
    model = unet.MultiBandUNet()
    model.load_state_dict(torch.load(args.model))
    if args.gpu >= 0:
        model.cuda()
    print('done')

    print('loading wave source...', end=' ')
    X, sr = librosa.load(
        args.input, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
    print('done')

    print('wave source stft...', end=' ')
    X, phase = spec_utils.calc_spec(X, args.hop_length, phase=True)
    coeff = X.max()
    X /= coeff
    print('done')

    left = model.offset
    roi_size = args.window_size - left * 2
    right = roi_size - (X.shape[2] % roi_size) + left
    X_pad = np.pad(X, ((0, 0), (0, 0), (left, right)), mode='reflect')

    masks = []
    model.eval()
    with torch.no_grad():
        for j in tqdm(range(int(np.ceil(X.shape[2] / roi_size)))):
            start = j * roi_size
            X_window = X_pad[None, :, :, start:start + args.window_size]
            X_tta = np.concatenate([X_window, X_window[:, ::-1, :, :]])

            pred = model(torch.from_numpy(X_tta).cuda())
            pred = pred.detach().cpu().numpy()
            pred[1] = pred[1, ::-1, :, :]
            masks.append(pred.mean(axis=0))

    mask = np.concatenate(masks, axis=2)[:, :, :X.shape[2]]
    if args.postprocess:
        vocal_pred = X * (1 - mask) * coeff
        mask = spec_utils.mask_uninformative(mask, vocal_pred)
    inst_pred = X * mask * coeff
    vocal_pred = X * (1 - mask) * coeff

    if args.out_mask:
        norm_mask = np.uint8(mask.mean(axis=0) * 255)[::-1]
        hm = cv2.applyColorMap(norm_mask, cv2.COLORMAP_MAGMA)
        cv2.imwrite('mask.png', hm)

    print('instrumental inverse stft...', end=' ')
    wav = spec_utils.spec_to_wav(inst_pred, phase, args.hop_length)
    print('done')
    librosa.output.write_wav('instrumental.wav', wav, sr)

    print('vocal inverse stft...', end=' ')
    wav = spec_utils.spec_to_wav(vocal_pred, phase, args.hop_length)
    print('done')
    librosa.output.write_wav('vocal.wav', wav, sr)


if __name__ == '__main__':
    main()
