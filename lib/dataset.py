import os

import numpy as np
import torch
from tqdm import tqdm

from lib import spec_utils


class VocalRemoverValidationSet(torch.utils.data.Dataset):

    def __init__(self, filelist):
        self.filelist = filelist

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        path = self.filelist[idx]
        data = np.load(path)

        return data['X'], data['y']


def mixup_generator(X, y, alpha):
    perm = np.random.permutation(len(X))[:len(X) // 2]
    if len(perm) % 2 != 0:
        perm = perm[:-1]
    for i in range(0, len(perm), 2):
        lam = np.random.beta(alpha, alpha)
        X[perm[i]] = lam * X[perm[i]] + (1 - lam) * X[perm[i + 1]]
        y[perm[i]] = lam * y[perm[i]] + (1 - lam) * y[perm[i + 1]]

    return X, y


def get_oracle_data(X, y, instance_loss, oracle_rate, oracle_drop_rate):
    k = int(len(X) * oracle_rate * (1 / (1 - oracle_drop_rate)))
    n = int(len(X) * oracle_rate)
    idx = np.argsort(instance_loss)[::-1][:k]
    idx = np.random.choice(idx, n, replace=False)
    oracle_X = X[idx].copy()
    oracle_y = y[idx].copy()

    return oracle_X, oracle_y, idx


def make_training_set(filelist, cropsize, patches, sr, hop_length):
    len_dataset = patches * len(filelist)
    X_dataset = np.zeros(
        (len_dataset, 2, hop_length, cropsize), dtype=np.float32)
    y_dataset = np.zeros(
        (len_dataset, 2, hop_length, cropsize), dtype=np.float32)
    for i, (X_path, y_path) in enumerate(tqdm(filelist)):
        X, y = spec_utils.cache_or_load(X_path, y_path, sr, hop_length)
        for j in range(patches):
            idx = i * patches + j
            start = np.random.randint(0, X.shape[2] - cropsize)
            X_dataset[idx] = X[:, :, start:start + cropsize]
            y_dataset[idx] = y[:, :, start:start + cropsize]
            if np.random.uniform() < 0.5:
                # swap channel
                X_dataset[idx] = X_dataset[idx, ::-1]
                y_dataset[idx] = y_dataset[idx, ::-1]

    return X_dataset, y_dataset


def make_validation_set(filelist, cropsize, offset, sr, hop_length, outdir='./val_patches', skip_if_exists=True):
    patch_list = []
    os.makedirs(outdir, exist_ok=True)
    for i, (X_path, y_path) in enumerate(tqdm(filelist)):
        X, y = spec_utils.cache_or_load(X_path, y_path, sr, hop_length)
        left = offset
        roi_size = cropsize - left * 2
        right = roi_size - (X.shape[2] % roi_size) + left
        X_pad = np.pad(X, ((0, 0), (0, 0), (left, right)), mode='constant')
        y_pad = np.pad(y, ((0, 0), (0, 0), (left, right)), mode='constant')
        len_dataset = int(np.ceil(X.shape[2] / roi_size))
        for j in range(len_dataset):
            outpath = os.path.join(outdir, '{:04}_p{:03}.npz'.format(i, j))
            start = j * roi_size
            if not os.path.exists(outpath) or not skip_if_exists:
                np.savez(
                    outpath,
                    X=X_pad[:, :, start:start + cropsize],
                    y=y_pad[:, :, start:start + cropsize])
            patch_list.append(outpath)

    return VocalRemoverValidationSet(patch_list)
