import numpy as np
from tqdm import tqdm

from lib import spec_utils


def mixup_generator(X, y, alpha):
    perm = np.random.permutation(len(X))[:len(X) // 2]
    if len(perm) % 2 != 0:
        perm = perm[:-1]
    for i in range(0, len(perm), 2):
        lam = np.random.beta(alpha, alpha)
        X[perm[i]] = lam * X[perm[i]] + (1 - lam) * X[perm[i + 1]]
        y[perm[i]] = lam * y[perm[i]] + (1 - lam) * y[perm[i + 1]]

    return X, y


def create_dataset(filelist, cropsize, patches=32, validation=False):
    len_dataset = patches * len(filelist)
    X_dataset = np.zeros(
        (len_dataset, 2, cropsize, cropsize), dtype=np.float32)
    y_dataset = np.zeros(
        (len_dataset, 2, cropsize, cropsize), dtype=np.float32)
    for i, (X_path, y_path) in enumerate(tqdm(filelist)):
        X, y = spec_utils.cache_or_load(X_path, y_path)
        for j in range(patches):
            idx = i * patches + j
            start = np.random.randint(0, X.shape[2] - cropsize)
            X_dataset[idx] = X[:, :, start:start + cropsize]
            y_dataset[idx] = y[:, :, start:start + cropsize]
            if not validation:
                if np.random.uniform() < 0.5:
                    # flip time
                    X_dataset[idx] = X_dataset[idx, :, :, ::-1]
                    y_dataset[idx] = y_dataset[idx, :, :, ::-1]
                if np.random.uniform() < 0.5:
                    # flip lr
                    X_dataset[idx] = X_dataset[idx, ::-1]
                    y_dataset[idx] = y_dataset[idx, ::-1]
                # elif np.random.uniform() < 0.2:
                #     # mono
                #     channel = np.random.randint(0, 1 + 1)
                #     X_dataset[idx, channel] = X_dataset[idx, channel - 1]
                #     y_dataset[idx, channel] = y_dataset[idx, channel - 1]

    return X_dataset, y_dataset
