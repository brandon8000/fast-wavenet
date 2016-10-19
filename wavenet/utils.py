import numpy as np
from scipy.io import wavfile

def normalize(data):
    temp = np.float32(data) - np.average(data)
    out = temp / np.max(np.abs(temp))
    return out


def make_batch(path):
    rate, data = wavfile.read(path)
    data = data[:, 0]
    data_ = normalize(data)
    bins, bins_center = mu_law_bins(256)
    inputs = np.digitize(data_[0:-1], bins, right=False)
    inputs = bins_center[inputs][None, :, None]
    targets = np.digitize(data_[1::], bins, right=False)[None, :]
    return (inputs, targets)


def make_batch_padded(path, num_layers = 14):
    rate, data = wavfile.read(path)
    data = data[:, 0]
    data_ = normalize(data)
    bins, bins_center = mu_law_bins(256)
    inputs = np.digitize(data_[0:-1], bins, right=False)
    inputs = bins_center[inputs][None, :, None]
    targets = np.digitize(data_[1::], bins, right=False)[None, :]
    base = 2 ** num_layers
    _, width, _ = inputs.shape
    width_cropped = int(np.floor(width * 1.0 / base) * base)
    inputs_padded = np.pad(inputs[:, 0:width_cropped, :], ((0, 0), (base - 1, 0), (0, 0)), 'constant')
    targets_padded = targets[:, 0:width_cropped]
    return (inputs_padded, targets_padded)


def mu_law_bins(num_bins):
    """ this functions returns the mu-law bin (right) edges and bin centers, with num_bins number of bins """
    bins_edge = np.linspace(-1, 1, num_bins + 1)
    bins_center = np.linspace(-1 + 1.0 / num_bins, 1 - 1.0 / num_bins, num_bins)
    bins_trunc = bins_edge[1:]
    bins_trunc[-1] += 0.1
    bins_edge_mu = np.multiply(np.sign(bins_trunc), (num_bins ** np.absolute(bins_trunc) - 1) / (num_bins - 1))
    bins_center_mu = np.multiply(np.sign(bins_center), (num_bins ** np.absolute(bins_center) - 1) / (num_bins - 1))
    return (bins_edge_mu, bins_center_mu)
