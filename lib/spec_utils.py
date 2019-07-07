import os

import chainer.functions as F
import librosa
import numpy as np


def crop_and_concat(h1, h2, concat=True):
    # s_freq = (h2.shape[2] - h1.shape[2]) // 2
    # e_freq = s_freq + h1.shape[2]
    s_time = (h2.shape[3] - h1.shape[3]) // 2
    e_time = s_time + h1.shape[3]
    h2 = h2[:, :, :, s_time:e_time]
    if concat:
        return F.concat([h1, h2])
    else:
        return h2


def calc_spec(X, phase=False):
    spec_left = librosa.stft(X[0], 1022, hop_length=512)
    mag_left = np.abs(spec_left)

    spec_right = librosa.stft(X[1], 1022, hop_length=512)
    mag_right = np.abs(spec_right)

    mag = np.asarray([mag_left, mag_right])

    if phase:
        phase_left = np.exp(1.j * np.angle(spec_left))
        phase_right = np.exp(1.j * np.angle(spec_right))
        phase = np.asarray([phase_left, phase_right])
        return mag, phase
    else:
        return mag


def cache_or_load(mix_path, inst_path, sr=32000):
    _, ext = os.path.splitext(mix_path)
    spec_mix_path = mix_path.replace(ext, '.npy')
    spec_inst_path = inst_path.replace(ext, '.npy')

    if os.path.exists(spec_mix_path) and os.path.exists(spec_inst_path):
        X = np.load(spec_mix_path)
        y = np.load(spec_inst_path)
    else:
        X, _ = librosa.load(
            mix_path, sr, False, dtype=np.float32, res_type='kaiser_fast')
        y, _ = librosa.load(
            inst_path, sr, False, dtype=np.float32, res_type='kaiser_fast')
        X, _ = librosa.effects.trim(X)
        y, _ = librosa.effects.trim(y)

        X_mono = X[:, :sr * 2].sum(axis=0)
        y_mono = y[:, :sr * 2].sum(axis=0)
        X_mono -= X_mono.mean()
        y_mono -= y_mono.mean()
        offset = len(y_mono) - 1
        delay = np.argmax(np.correlate(X_mono, y_mono, 'full')) - offset

        if delay > 0:
            X = X[:, delay:]
        else:
            y = y[:, np.abs(delay):]
        if X.shape[1] < y.shape[1]:
            y = y[:, :X.shape[1]]
        else:
            X = X[:, :y.shape[1]]

        # if 'mixture' in mix_path and delay != 0:
        #     print('\n', mix_path, delay)
        #     librosa.output.write_wav('noisy.wav', X - y, sr * 2)
        #     input()

        X = calc_spec(X)
        y = calc_spec(y)

        _, ext = os.path.splitext(mix_path)
        np.save(spec_mix_path, X)
        np.save(spec_inst_path, y)

    coeff = np.max([X.max(), y.max()])
    return X / coeff, y / coeff


def spec_to_wav(spec, phase, hop_length, ref_max):
    offset = phase.shape[2]
    spec = spec[:, :, :offset] * ref_max * phase
    wav_left = librosa.istft(spec[0], hop_length=hop_length)
    wav_right = librosa.istft(spec[1], hop_length=hop_length)
    wav = np.asarray([wav_left, wav_right])
    return wav
