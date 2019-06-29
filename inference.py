import argparse

import chainer
from chainer import backends
import librosa
import numpy as np
from tqdm import tqdm

from lib import spec_utils
from lib import unet


p = argparse.ArgumentParser()
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--model', '-m', type=str, default='models/baseline.npz')
p.add_argument('--input', '-i', required=True)
p.add_argument('--window_size', '-w', type=int, default=1024)
args = p.parse_args()


if __name__ == '__main__':
    model = unet.SpecUNet()
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu()

    print('loading raw wave form...', end=' ')
    X, _ = librosa.load(args.input, 44100, mono=False, dtype=np.float32)
    print('done')

    print('mixture stft...', end=' ')
    X, phase = spec_utils.calc_spec(X, True)
    print('done')

    ref_max = X.max()
    X /= ref_max

    left = model.offset
    roi_size = args.window_size - left * 2
    right = roi_size + left - (X.shape[2] % left)
    X_pad = np.pad(X, ((0, 0), (0, 0), (left, right)), mode='edge')

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
            mask = model(model.xp.asarray(X_tta))
            mask = backends.cuda.to_cpu(mask.data)
            mask[1] = mask[1, :, :, ::-1]
            mask[2] = mask[2, ::-1, :, :]
            mask[3] = mask[3, ::-1, :, ::-1]
            mask = mask.mean(axis=0)[None]
            X_window = spec_utils.crop_and_concat(mask, X_window, False)
            inst_preds.append((X_window * mask)[0])
            vocal_preds.append((X_window * (1 - mask))[0])

    inst_preds = np.concatenate(inst_preds, axis=2)
    print('instrumental inverse stft...', end=' ')
    wav = spec_utils.spec_to_wav(inst_preds, phase, 512, ref_max)
    print('done')
    librosa.output.write_wav('instrumental.wav', wav, 44100)

    vocal_preds = np.concatenate(vocal_preds, axis=2)
    print('vocal inverse stft...', end=' ')
    wav = spec_utils.spec_to_wav(vocal_preds, phase, 512, ref_max)
    print('done')
    librosa.output.write_wav('vocal.wav', wav, 44100)
