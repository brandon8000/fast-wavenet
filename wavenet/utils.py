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
    # data_f = np.sign(data_) * (np.log(1 + 255*np.abs(data_)) / np.log(1 + 255))

    bins,bins_center = mu_law_bins(256)
    # Quantize inputs.
    inputs = np.digitize(data_[0:-1], bins, right=False)
    inputs = bins_center[inputs][None, :, None]

    # Encode targets as ints.
    targets = np.digitize(data_[1::], bins, right=False)[None, :]
    
    return inputs, targets

def mu_law_bins(num_bins):
    ''' this functions returns the mu-law bin (right) edges and bin centers, with num_bins number of bins '''
    bins_edge = np.linspace(-1, 1, num_bins + 1)
    bins_center = np.linspace(-1 + 1.0 / num_bins, 1 - 1.0 / num_bins, num_bins)
    bins_trunc = bins_edge[1:] # remove left edge
    
    # slightly up adjust the last bin edge to avoid exact equal
    bins_trunc[-1] += 0.1
    
    # apply inverse mu-law companding
    bins_edge_mu = np.multiply(np.sign(bins_trunc),
                               (num_bins ** np.absolute(bins_trunc) - 1) / (num_bins - 1))
    bins_center_mu = np.multiply(np.sign(bins_center), 
                               (num_bins ** np.absolute(bins_center) - 1) / (num_bins - 1))
    
    return bins_edge_mu, bins_center_mu