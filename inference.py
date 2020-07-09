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


def reconstruct(X, window_size, model, device, roll=False):
    l, r, roi_size = dataset.make_padding(X.shape[2], window_size, model.offset)
    X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode='constant')

    if roll:
        X_pad = np.roll(X_pad, roi_size // 2, axis=2)

    model.eval()
    with torch.no_grad():
        preds = []
        for i in tqdm(range(int(np.ceil(X.shape[2] / roi_size)))):
            start = i * roi_size
            X_window = torch.from_numpy(np.asarray([
                X_pad[:, :, start:start + window_size],
            ])).to(device)

            pred = model.predict(X_window)

            pred = pred.detach().cpu().numpy()
            preds.append(pred[0])

        pred = np.concatenate(preds, axis=2)[:, :, :X.shape[2]]

    if roll:
        pred = np.roll(pred, -roi_size // 2, axis=2)

    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default='models/baseline.pth')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-l', type=int, default=1024)
    p.add_argument('--window_size', '-w', type=int, default=512)
    p.add_argument('--output_image', '-I', action='store_true')
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    model = nets.CascadedASPPNet(args.n_fft)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)
    print('done')

    print('loading wave source...', end=' ')
    X, sr = librosa.load(
        args.input, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
    basename = os.path.splitext(os.path.basename(args.input))[0]
    print('done')

    print('stft of wave source...', end=' ')
    X = spec_utils.get_spectrogram(X, args.hop_length, args.n_fft)
    X_mag, X_phase = np.abs(X), np.angle(X)
    print('done')

    pred = reconstruct(X_mag, args.window_size, model, device)
    pred_roll = reconstruct(X_mag, args.window_size, model, device, roll=True)
    pred = (pred + pred_roll) / 2

    print('inverse stft of instruments...', end=' ')
    y_spec = pred * np.exp(1.j * X_phase)
    wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}_Instruments.wav'.format(basename), wave.T, sr)

    print('inverse stft of vocals...', end=' ')
    v_spec = np.clip(X_mag - pred, 0, np.inf) * np.exp(1.j * X_phase)
    wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}_Vocals.wav'.format(basename), wave.T, sr)

    if args.output_image:
        with open('{}_Instruments.jpg'.format(basename), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_Vocals.jpg'.format(basename), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(v_spec)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)


if __name__ == '__main__':
    main()
