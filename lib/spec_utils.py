import os

import librosa
import numpy as np
import soundfile as sf


def crop_center(h1, h2):
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError('h1_shape[3] must be greater than h2_shape[3]')

    # s_freq = (h2_shape[2] - h1_shape[2]) // 2
    # e_freq = s_freq + h1_shape[2]
    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    h1 = h1[:, :, :, s_time:e_time]

    return h1


def wave_to_spectrogram(wave, hop_length, n_fft):
    spec_left = librosa.stft(wave[0], n_fft=n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave[1], n_fft=n_fft, hop_length=hop_length)
    spec = np.asarray([spec_left, spec_right])

    return spec


def spectrogram_to_image(spec, mode='magnitude'):
    if mode == 'magnitude':
        if np.iscomplexobj(spec):
            y = np.abs(spec)
        else:
            y = spec
        y = np.log10(y ** 2 + 1e-8)
    elif mode == 'phase':
        if np.iscomplexobj(spec):
            y = np.angle(spec)
        else:
            y = spec

    y -= y.min()
    y *= 255 / y.max()
    img = np.uint8(y)

    if y.ndim == 3:
        img = img.transpose(1, 2, 0)
        img = np.concatenate([
            np.max(img, axis=2, keepdims=True), img
        ], axis=2)

    return img


def get_reduction_weight(n_fft, sr, reduction_level):
    bins = n_fft // 2 + 1
    freq_to_bin = 2 * bins / sr
    unstable_bins = int(200 * freq_to_bin)
    stable_bins = int(22050 * freq_to_bin)
    return np.concatenate([
        np.linspace(0, 1, unstable_bins + 1, dtype=np.float32)[:unstable_bins, None],
        np.linspace(1, 0, stable_bins - unstable_bins, dtype=np.float32)[:, None],
        np.zeros((bins - stable_bins, 1), dtype=np.float32),
    ], axis=0) * reduction_level


def align_wave_head_and_tail(a, b, sr):
    a, _ = librosa.effects.trim(a)
    b, _ = librosa.effects.trim(b)

    a_mono = a[:, :sr * 4].sum(axis=0)
    b_mono = b[:, :sr * 4].sum(axis=0)

    a_mono -= a_mono.mean()
    b_mono -= b_mono.mean()

    offset = len(a_mono) - 1
    delay = np.argmax(np.correlate(a_mono, b_mono, 'full')) - offset

    if delay > 0:
        a = a[:, delay:]
    else:
        b = b[:, np.abs(delay):]

    if a.shape[1] < b.shape[1]:
        b = b[:, :a.shape[1]]
    else:
        a = a[:, :b.shape[1]]

    return a, b


def cache_or_load(X_path, y_path, v_path, sr, hop_length, n_fft):
    X_basename = os.path.splitext(os.path.basename(X_path))[0]
    y_basename = os.path.splitext(os.path.basename(y_path))[0]
    v_basename = os.path.splitext(os.path.basename(v_path))[0]

    cache_dir = 'sr{}_hl{}_nf{}'.format(sr, hop_length, n_fft)
    X_cache_dir = os.path.join(os.path.dirname(X_path), cache_dir)
    y_cache_dir = os.path.join(os.path.dirname(y_path), cache_dir)
    v_cache_dir = os.path.join(os.path.dirname(v_path), cache_dir)

    X_cache_path = os.path.join(X_cache_dir, X_basename + '.npy')
    y_cache_path = os.path.join(y_cache_dir, y_basename + '.npy')
    v_cache_path = os.path.join(v_cache_dir, v_basename + '.npy')

    if os.path.exists(X_cache_path) and os.path.exists(y_cache_path) and os.path.exists(v_cache_path):
        X = np.load(X_cache_path).transpose(1, 2, 0)
        y = np.load(y_cache_path).transpose(1, 2, 0)
        v = np.load(v_cache_path).transpose(1, 2, 0)

    assert X.shape == y.shape == v.shape

    return X, y, v, X_cache_path, y_cache_path, v_cache_path


def spectrogram_to_wave(spec, hop_length=1024):
    if spec.ndim == 2:
        wave = librosa.istft(spec, hop_length=hop_length)
    elif spec.ndim == 3:
        wave_left = librosa.istft(spec[0], hop_length=hop_length)
        wave_right = librosa.istft(spec[1], hop_length=hop_length)
        wave = np.asarray([wave_left, wave_right])

    return wave


if __name__ == "__main__":
    import cv2
    import sys

    X, _ = librosa.load(
        sys.argv[1], sr=44100, mono=False, dtype=np.float32, res_type='kaiser_fast'
    )
    y, _ = librosa.load(
        sys.argv[2], sr=44100, mono=False, dtype=np.float32, res_type='kaiser_fast'
    )

    X, y = align_wave_head_and_tail(X, y, 44100)
    X_spec = wave_to_spectrogram(X, 1024, 2048)
    y_spec = wave_to_spectrogram(y, 1024, 2048)

    # X_spec = np.load(sys.argv[1]).transpose(1, 2, 0)
    # y_spec = np.load(sys.argv[2]).transpose(1, 2, 0)

    v_spec = X_spec - y_spec

    X_image = spectrogram_to_image(X_spec)
    y_image = spectrogram_to_image(y_spec)
    v_image = spectrogram_to_image(v_spec)

    cv2.imwrite('test_X.jpg', X_image)
    cv2.imwrite('test_y.jpg', y_image)
    cv2.imwrite('test_v.jpg', v_image)

    sf.write('test_X.wav', spectrogram_to_wave(X_spec).T, 44100)
    sf.write('test_y.wav', spectrogram_to_wave(y_spec).T, 44100)
    sf.write('test_v.wav', spectrogram_to_wave(v_spec).T, 44100)
