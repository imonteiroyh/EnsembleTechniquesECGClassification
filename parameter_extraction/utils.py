import math
import numpy as np
from scipy.signal import filtfilt, convolve, hilbert, find_peaks

# FILTERS
def decomposition_baseline_filter(lo_D, hi_D, cA, f_samp, n):
    A_list = []
    for i in range(0, n):
        # Convolution without signal offset and signal extension in the beggining and in the end of the signal
        A = 0.5*filtfilt(lo_D, 1, cA, padlen=0)
        A = np.concatenate((A[0]*np.ones((1,f_samp))[0], A, A[-1]*np.ones((1,f_samp))[0]), axis=None)
        A_list.append(A)

        D = 0.5*filtfilt(hi_D, 1, cA, padlen=0)
        D = np.concatenate((D[0]*np.ones((1,f_samp))[0], D, D[-1]*np.ones((1,f_samp))[0]), axis=None)

        # Signal downsampling excluding half of the samples
        cA = [x for index, x in enumerate(A) if index%2 == 0]
        cD = [x for index, x in enumerate(D) if index%2 == 0]

    return cA, cD, A_list

def recomposition_baseline_filter(A_list, lo_R, hi_R, cA, cD, f_samp, n):
    for i in range(0,n):
        # Signal interleaving with zeros
        scA = np.zeros((1, 2*len(cA)))[0]
        scA = [cA[int(index/2)] if index%2==0 else 0 for index, x in enumerate(scA)]
        scD = np.zeros((1, 2*len(cD)))[0]
        scD = [cD[int(index/2)] if index%2==0 else 0 for index, x in enumerate(scD)]

        if len(scA) > len(A_list[n-1-i]):
            scA = scA[0:len(scA)-1]
            scD = scD[0:len(scD)-1]

        # Applying the filter and removing samples in the beggining and in the end of the signal
        cA = filtfilt(lo_R,1,scA, padlen=0)
        cA = cA[f_samp:len(cA)-f_samp]

    return cA

def baseline_filter(data, f_samp, n):
    hi_D = [-0.2304, 0.7148, -0.6309, -0.0280, 0.1870, 0.0308, -0.0329, -0.0106]
    lo_D = [-0.0106, 0.0329, 0.0308, -0.1870, -0.0280, 0.6309, 0.7148, 0.2304]
    hi_R = [-0.0106, -0.0329, 0.0308, 0.1870, -0.0280, -0.6309, 0.7148, -0.2304]
    lo_R = [0.2304, 0.7148, 0.6309, -0.0280, -0.1870, 0.0308, 0.0329, -0.0106]

    cA, cD, A_list = decomposition_baseline_filter(lo_D, hi_D, data, f_samp, n)
    cA = recomposition_baseline_filter(A_list, lo_R, hi_R, cA, cD, f_samp, n)

    filtered_data = data - cA

    return filtered_data

def generate_mexican_hat_filter(scale = 2):
    step = 1/scale
    X = np.arange(-5, 5, step)
    np.append(X, 5)

    Y = []
    for i in range(0, len(X)):
        Y.append(2.1741*(1/math.sqrt(2*math.pi) * (1 - X[i]**2) * np.exp(-X[i]**2/2)))

    return Y

# TRANSFORMATIONS
def get_superenergy_signal(data, f_samp, scale = 2):
    filter = generate_mexican_hat_filter(scale)
    window_size = round(0.15*f_samp)

    samples_lenght = len(data[0])
    result_signal = np.zeros(samples_lenght)

    for iter_channel in range(12):
        filtered_data = convolve(data[iter_channel], filter)

        # Removal of non-useful data resulting from convolution
        gap = int(np.round(len(filter)/2)) - 1
        filtered_data = np.copy(filtered_data[gap : (len(filtered_data) - gap)])

        derivative_filter = np.diff(filtered_data)

        envelope = hilbert(derivative_filter)
        envelope_amplitude = np.abs(envelope)

        result_signal = result_signal + envelope_amplitude

    result_signal_index = list(range(len(result_signal)))
    result_signal = np.array([0 if (index < window_size or index > len(result_signal) - (window_size + 1)) else signal for index, signal in zip(result_signal_index, result_signal)])

    return result_signal

def get_superenergy_signal_peaks(superenergy_signal, initial_threshold, minimum_interpeak_distance, f_samp):
    start_position = round(0.04 * f_samp)
    end_position = len(superenergy_signal) - round(0.2 * f_samp)

    min_number_of_beats = np.floor(0.7 * len(superenergy_signal) / f_samp)
    max_number_of_beats =round(3.5 * len(superenergy_signal) / f_samp)

    threshold = initial_threshold
    superenergy_signal_max_amplitude = max(abs(superenergy_signal[start_position : end_position + 1]))
    normalized_superenergy_signal = superenergy_signal / superenergy_signal_max_amplitude

    iter = 0
    stop = False
    while stop == False and iter < 50:
        signal_threshold = threshold * max(abs(normalized_superenergy_signal))

        try:
            superenergy_signal_peaks = find_peaks(normalized_superenergy_signal, height = signal_threshold, distance = minimum_interpeak_distance)[0]
        except:
            superenergy_signal_peaks = []

        if len(superenergy_signal_peaks) >= min_number_of_beats and len(superenergy_signal_peaks) < max_number_of_beats:
            stop = True
        else:
            if len(superenergy_signal_peaks) < min_number_of_beats and threshold >= 0.1 * initial_threshold:
                threshold = 0.8 * threshold
            else:
                threshold = 1.2 * threshold

        iter += 1

    return superenergy_signal_peaks

def wavelet_transform(channel_data, num):
    gn = [-2, 2]
    hn = [1/8, 3/8, 3/8, 1/8]

    for iter in range(num):
        wavelet_transformed_data = convolve(channel_data, gn)

        for iter_gn in range(int(len(gn)/2)):
            wavelet_transformed_data = np.delete(wavelet_transformed_data, 0)
            wavelet_transformed_data = np.delete(wavelet_transformed_data, len(wavelet_transformed_data) - 1)

        channel_data = convolve(channel_data, hn)

        for iter_hn in range(int(len(hn)/2)):
            channel_data = np.delete(channel_data, 0)
            channel_data = np.delete(channel_data, len(channel_data) - 1)

        new_gn = []
        new_hn = []

        for iter_gn in range(len(gn)):
            new_gn.append(gn[iter_gn])
            new_gn.append(0)
        gn = new_gn

        for iter_hn in range(len(hn)):
            new_hn.append(hn[iter_hn])
            new_hn.append(0)
        hn = new_hn

    return wavelet_transformed_data

def get_acl(wavelet_transformed_data, floating_window_size):
    acl = np.zeros(len(wavelet_transformed_data))

    '''
    The ACL metric is defined by the product of the function relative to the area under the wave and the function
    representing the curve of the wave
    '''

    for iter_sample in range(len(wavelet_transformed_data) - floating_window_size):

        # y_k is a vector including samples from k to k + L of the filtered version relative to scale 2λ
        y_k = wavelet_transformed_data[iter_sample : (iter_sample + floating_window_size - 1)]
        area_k = sum(abs(y_k))

        curve_k = 0
        for iter_yk in range(1, len(y_k)):
            curve_k = curve_k + math.sqrt(1 + (y_k[iter_yk] - y_k[iter_yk - 1])**2)

        acl[iter_sample] = area_k*curve_k

    return acl