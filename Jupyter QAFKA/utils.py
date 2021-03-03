import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import geom, nbinom

def clean_bg_noise(data, patch_length):
    [N, H, W] = data.shape
    clean_data = np.zeros_like(data)
    half_size = int(np.floor(patch_length/2))
    for i in range(H):
        for j in range(W):
            up = np.min([H - 1, i + half_size])
            down = np.max([0, i - half_size])
            right = np.min([W - 1, j + half_size])
            left = np.max([0, j - half_size])
            clean_data[:, i, j] = data[:, i, j] - np.mean(data[:, down:up, left:right])
    print("-I- Background noise was filtered")
    return clean_data

def segment(data, threshold, window_size):
    ref_mean = np.mean(data)
    ref_std = np.std(data)

    for i in range(data.shape[0]-window_size):
        curr_mean = np.mean(data[i:i+window_size])
        curr_std = np.std(data[i:i+window_size])

        if((np.abs(curr_mean-ref_mean)/ref_mean + np.abs(curr_std-ref_std)/ref_std) < threshold):
            print("-I- Found segmentation in frame:", i)
            return i

    print("Error: could not segment, choosing frame 500 as segmentation frame")
    return 500

def calc_threshold(raw_data, max_data):
    """
        Calculating a "good" threshold fot peak detection
        :param raw_data: Tensor [image_size] of one experiment
        :return: float as the threshold.
    """
    return 2 * np.abs(np.min(raw_data)) + np.std(raw_data), 1.5 * np.abs(np.mean(np.median(max_data, axis=1), axis=1) + np.std(max_data))

def draw_circle(size, rad):
    # size should be odd
    circle = np.zeros([size, size], dtype=int)
    mid = int((size-1)/2)
    for i in range(size):
        for j in range(size):
            if((i-mid)**2 + (j-mid)**2 <= rad**2):
                circle[i, j] = 1

    return circle

def gauss2d(xy, offset, amp, x0, y0, sigma):
    # Fit patch to gaussian
    x, y = xy
    return offset + amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

def Normalization(X_train, X_val, X_test):
    """
        Normalizing the data by the statistics of X_train
        :return: normalized X_train, X_val, X_test.
    """
    for i in range(X_train.shape[0]):
        X_train[i, :] /= np.sum(X_train[i])
    for i in range(X_val.shape[0]):
        X_val[i, :] /= np.sum(X_val[i])
    for i in range(X_test.shape[0]):
        X_test[i, :] /= np.sum(X_test[i])

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[np.where(std == 0)] = 1
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return torch.FloatTensor(X_train), torch.FloatTensor(X_val), torch.FloatTensor(X_test)

def fit(n_blinks_hist):
    """
        Calculates the MLE for the percentage of dimers in an experiment
        :param n_blinks_hist: Tensor [numOfBIns] for the n_blinks histogram in an exp.
        :param p: Float for the bleaching probability of a cluster
        :return: alpha - the dimers percentage in the experiment.
    """
    best_alpha, best_p = -1, -1
    max_val = -np.inf
    for j in range(200, 500):
        p = j / 1000
        for i in range(101):
            alpha = i / 100
            val = 0
            for bin in range(1, len(n_blinks_hist) + 1):
                a = p * (1 - p) ** (bin-1)
                b = (bin * (p ** 2)) * (1 - p) ** (bin-1)
                val += n_blinks_hist[bin - 1] * np.log((1 - alpha) * a + alpha * b)

            if(val > max_val):
                max_val = val
                best_alpha = alpha
                best_p = p

    return best_alpha, best_p

def plot_n_blinks(n_blinks, bleach_proba):
    for i in range(n_blinks.shape[0]):
        plt.plot(n_blinks[i]/np.sum(n_blinks[i]), label='{}'.format(i))
    x1 = np.arange(geom.ppf(0.0001, bleach_proba), geom.ppf(0.9999, bleach_proba))
    x2 = np.arange(nbinom.ppf(0.0001, 2, bleach_proba), nbinom.ppf(0.9999, 2, bleach_proba))
    plt.plot(x1 - 1, geom.pmf(x1, bleach_proba), label='Monomers only')
    plt.plot(x2, nbinom.pmf(x2, 2, bleach_proba), label='Dimers only')
    plt.legend()
    plt.show()

def estimate_by_average(n_blinks_hist, bleach_proba):
    total_blinks = np.dot(n_blinks_hist, np.arange(1, 21))
    total_clusters = np.sum(n_blinks_hist)
    best_alpha = None
    best_diff = np.inf
    for i in range(101):
        alpha = i/100
        # Monomers part * Average monomers blinks + Dimers part * Average dimers blinks
        approx = ((1-alpha)*total_clusters)*((1/bleach_proba)) + \
                 (alpha*total_clusters)*((2*(1-bleach_proba)/bleach_proba)+1)
        diff = np.abs(approx - total_blinks)
        if(diff < best_diff):
            best_diff = diff
            best_alpha = alpha

    return best_alpha

def kmeans(data):
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    X_embedded = TSNE(n_components=2).fit_transform(data)
    nbrs = KMeans(n_clusters=2).fit(data)
    pred = nbrs.predict(data)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=pred)
    plt.show()

def plotting(hist, p):
    hist /= np.sum(hist)
    x1 = np.arange(geom.ppf(0.0001, p), geom.ppf(0.9999, p))
    n = 2
    x2 = np.arange(nbinom.ppf(0.001, n, p), nbinom.ppf(0.999, n, p))
    plt.plot(hist, label='Experiment histogram')
    plt.plot(x2 + 1, nbinom.pmf(x2, n, p), label='Dimers histogram')
    plt.plot(x1-1, geom.pmf(x1, p), label='Monomers histogram')
    plt.xticks(np.arange(20), np.arange(1, 21))
    plt.xlabel('Number of blinks')
    plt.ylabel('Probability')
    plt.title('Monomers and dimers Nblinks model')
    plt.legend()
    plt.show()

def CoordiantionsComparison(filename, myCoordination):
    data = np.loadtxt(filename, delimiter=' ')
    col = data[:, 0] / 40200
    row = data[:, 1] / 40270
    plt.scatter(row, col)
    plt.scatter(myCoordination[:, 0]/251, myCoordination[:, 1]/257)
    plt.xlabel('x [normalized units]')
    plt.ylabel('y [normalized units]')
    plt.legend(['Tims localizations', 'My localizations'])
    plt.show()

def debug_helper(data, emitters_grid, fit, obs, center_x, center_y, fit_quality, patch_length):
    plt.gca().invert_yaxis()
    plt.subplot(221)
    plt.imshow(data)
    plt.plot(center_y, center_x, color='r', marker='x')
    plt.subplot(222)
    plt.imshow(emitters_grid)
    plt.subplot(223)
    plt.title("Quality {}".format(fit_quality))
    plt.imshow(fit.reshape([patch_length, patch_length]))
    plt.subplot(224)
    plt.title("Localization {}, {}".format(center_x, center_y))
    plt.imshow(obs.reshape([patch_length, patch_length]))
    plt.show()

def MLE(n_blinks_hist, p, d):
    """
        Calculates the MLE for the percentage of dimers in an experiment
        :param n_blinks_hist: Tensor [numOfBIns] for the n_blinks histogram in an exp.
        :param p: Float for the bleaching probability of a cluster
        :return: alpha - the dimers percentage in the experiment.
    """
    best_alpha = -1
    max_val = -np.inf

    val_vec = np.zeros(100)
    hist_vec = np.ones(int(np.sum(n_blinks_hist)))
    for i in range(len(n_blinks_hist) - 1):
        hist_vec[int(np.sum(n_blinks_hist[:i])):int(np.sum(n_blinks_hist[:i+1]))] = i + 1

    for i in range(101):
        alpha = i / 100
        val = 0
        for n in hist_vec:
            if(n > 1):
                a = p * (1 - p) ** (n - 1)
                b = (d * ((n - 1) * (p ** 2)) * (1 - p) ** (n - 2) + (1-d) * p * (1 - p) ** (n-1))
            else:
                a = p * (1 - p) ** (n - 1)
                b = (1-d) * p * (1 - p) ** (n-1)

            val += np.log((1 - alpha) * a + alpha * b)

        if(val > max_val):
            max_val = val
            best_alpha = alpha
        val_vec[i-1] = val
    plt.plot(val_vec)
    plt.title("Real dimers percentage: {}".format(best_alpha))
    plt.show()
    return best_alpha

def find_atual_dimers_percentage(m, d):
    real_percentage = 1/(((1-m)/m + 2) * d - 1)
    return real_percentage

def debug_entire_exp(Max_Data_Set, coordinates, scale_size):
    new_img = np.zeros([scale_size * Max_Data_Set[1].shape[0], scale_size * Max_Data_Set[1].shape[1]])
    for i in range(Max_Data_Set[1].shape[0] - 1):
        for j in range(Max_Data_Set[1].shape[1] - 1):
            new_img[(scale_size * i):(scale_size * (i + 1)), (scale_size * j):(scale_size * (j + 1))] = np.max(Max_Data_Set[:, i, j])
    plt.figure(figsize=(10,10))
    plt.imshow(new_img)
    plt.scatter(coordinates[:, 1], coordinates[:, 0], color='r', marker='x')
    plt.show()

'''m = 0.52
q = 0.36
d = (2-2*q)/(2-q)
monomers = geom.pmf(np.arange(1,21), 0.41)
dimers = np.zeros(20)
dimers[1:] = nbinom.pmf(np.arange(19), 2, 0.41)
data = (1-m) * 10000 * monomers + (m) * 10000 * dimers

MLE(data, 0.41, d)
find_atual_dimers_percentage(m, d)
'''