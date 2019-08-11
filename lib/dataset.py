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


def get_oracle_data(X, y, instance_loss, oracle_rate, oracle_drop_rate):
    k = int(len(X) * oracle_rate * (1 / (1 - oracle_drop_rate)))
    n = int(len(X) * oracle_rate)
    idx = np.argsort(instance_loss)[::-1][:k]
    idx = np.random.choice(idx, n, replace=False)
    oracle_X = X[idx].copy()
    oracle_y = y[idx].copy()
    return oracle_X, oracle_y, idx


def create_dataset(filelist, cropsize, patches, sr, hop_length,
                   validation=False):
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
            if not validation:
                if np.random.uniform() < 0.5:
                    # swap channel
                    X_dataset[idx] = X_dataset[idx, ::-1]
                    y_dataset[idx] = y_dataset[idx, ::-1]
                # if np.random.uniform() < 0.5:
                #     f = np.random.randint(0, 512 // 4)
                #     f0 = np.random.randint(0, 512 - f)
                #     X_dataset[idx, :, f0:f0 + f, :] = 0
                #     y_dataset[idx, :, f0:f0 + f, :] = 0

    return X_dataset, y_dataset
