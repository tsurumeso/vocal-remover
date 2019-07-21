import argparse

import chainer
from chainer import backends
import cv2
import librosa
import numpy as np
from tqdm import tqdm

from lib import spec_utils
from lib import unet


p = argparse.ArgumentParser()
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--model', '-m', type=str, default='models/baseline.npz')
p.add_argument('--input', '-i', required=True)
p.add_argument('--sr', '-r', type=int, default=44100)
p.add_argument('--hop_length', '-l', type=int, default=1024)
p.add_argument('--window_size', '-w', type=int, default=1024)
args = p.parse_args()


if __name__ == '__main__':
    print('loading model...', end=' ')
    model = unet.MultiBandUNet()
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu()
    print('done')

    print('loading wave source...', end=' ')
    X, sr = librosa.load(
        args.input, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
    print('done')

    print('wave source stft...', end=' ')
    X, phase = spec_utils.calc_spec(X, args.hop_length, phase=True)
    ref_max = X.max()
    X /= ref_max
    print('done')

    left = model.offset
    roi_size = args.window_size - left * 2
    right = roi_size + left - (X.shape[2] % left)
    X_pad = np.pad(X, ((0, 0), (0, 0), (left, right)), mode='reflect')

    masks = []
    inst_preds = []
    vocal_preds = []
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        for j in tqdm(range(int(np.ceil(X.shape[2] / roi_size)))):
            start = j * roi_size
            X_window = X_pad[None, :, :, start:start + args.window_size]
            X_tta = np.concatenate([
                X_window,
                X_window[:, :, :, ::-1],
                X_window[:, ::-1, :, :],
                X_window[:, ::-1, :, ::-1],
            ])

            pred = model(model.xp.asarray(X_tta))
            pred = backends.cuda.to_cpu(pred.data)
            pred[1] = pred[1, :, :, ::-1]
            pred[2] = pred[2, ::-1, :, :]
            pred[3] = pred[3, ::-1, :, ::-1]
            mask = pred.mean(axis=0)[None]

            norm_mask = np.uint8((np.mean(mask, axis=1)) * 255)[0, ::-1]
            hm = cv2.applyColorMap(norm_mask, cv2.COLORMAP_JET)
            masks.append(hm)

            X_window = spec_utils.crop_and_concat(mask, X_window, False)
            inst_preds.append((X_window * mask)[0])
            vocal_preds.append((X_window * (1 - mask))[0])

    mask = np.concatenate(masks, axis=1)
    cv2.imwrite('mask.png', mask)

    inst_preds = np.concatenate(inst_preds, axis=2)
    print('instrumental inverse stft...', end=' ')
    wav = spec_utils.spec_to_wav(inst_preds, phase, args.hop_length, ref_max)
    print('done')
    librosa.output.write_wav('instrumental.wav', wav, sr)

    vocal_preds = np.concatenate(vocal_preds, axis=2)
    print('vocal inverse stft...', end=' ')
    wav = spec_utils.spec_to_wav(vocal_preds, phase, args.hop_length, ref_max)
    print('done')
    librosa.output.write_wav('vocal.wav', wav, sr)
