import math
import numpy as np
from statistics import median_low, median_high
from parameter_extraction import utils

def get_qrs_peak(data, f_samp):
    qrs_peaks = []
    scales = [2, 3, 4]

    for scale in scales:
        superenergy_signal = utils.get_superenergy_signal(data, f_samp, scale)

        initial_threshold = 0.3
        minimum_interpeak_distance = 0.25 * f_samp
        superenergy_signal_peaks = utils.get_superenergy_signal_peaks(superenergy_signal, initial_threshold, minimum_interpeak_distance, f_samp)
        superenergy_signal_peaks_standard_deviation = np.std(superenergy_signal_peaks)

        current_scale_qrs_peaks = []
        qrs_peaks_total_standard_deviation = 0
        for iter_channel in range(12):
            current_channel_qrs_peaks = []
            current_channel_data = data[iter_channel]

            for iter_peak in range(len(superenergy_signal_peaks)):
                current_peak = superenergy_signal_peaks[iter_peak]

                # Searching the peak of the R wave in a 120 ms window
                window_size = round(0.12 * f_samp)

                if (current_peak - window_size) >= 0:
                    current_channel_qrs_peaks.append(current_peak - window_size + np.argmax(abs(current_channel_data[current_peak - window_size : current_peak + window_size])))
                else:
                    current_channel_qrs_peaks.append(np.argmax(abs(current_channel_data[0 : current_peak + window_size])))

            qrs_peaks_total_standard_deviation += np.std(current_channel_qrs_peaks)
            current_scale_qrs_peaks.append(current_channel_qrs_peaks)

        total_deviation = superenergy_signal_peaks_standard_deviation * qrs_peaks_total_standard_deviation
        if len(qrs_peaks) == 0 or total_deviation < best_total_deviation:
            best_total_deviation = total_deviation
            qrs_peaks = current_scale_qrs_peaks

    return qrs_peaks

def get_qrs_interval(qrs_peaks, acl, channel, f_samp, delay):
    qrs_on = []
    qrs_off = []

    for iter_peak in range(len(qrs_peaks[channel])):

        iter_sample_on = qrs_peaks[channel][iter_peak]
        if (iter_sample_on - round(0.12*f_samp) >= 1):
            window = acl[(iter_sample_on - round(0.12*f_samp)) : (iter_sample_on + round(0.12*f_samp))]
        else:
            window = acl[0 : (iter_sample_on + round(0.12*f_samp))]

        found = False
        while not(found):
            if acl[iter_sample_on] < 1.1*acl[iter_sample_on - 1] and acl[iter_sample_on] < 1.1*acl[iter_sample_on + 1] and qrs_peaks[channel][iter_peak] - (iter_sample_on + delay) >= 0.06*f_samp and acl[iter_sample_on] < 0.7*max(window):
                found = True
                break
            else:
                iter_sample_on = iter_sample_on - 1
        qrs_on.append(iter_sample_on + delay)

        iter_sample_end = qrs_peaks[channel][iter_peak]
        min_value = acl[iter_sample_end]
        min_position = iter_sample_end
        found = False
        while not(found):
            if acl[iter_sample_end] < 1.1*acl[iter_sample_end - 1] and acl[iter_sample_end] < 1.1*acl[iter_sample_end + 1] and iter_sample_end - qrs_peaks[channel][iter_peak] >= 0.06*f_samp and acl[iter_sample_end] < 0.7*max(window):
                found = True
                break
            else:
                if acl[iter_sample_end] <= min_value:
                    min_value = acl[iter_sample_end]
                    min_position = iter_sample_end
                iter_sample_end = iter_sample_end + 1

                if iter_sample_end >= len(acl) - 1:
                    iter_sample_end = min_position
                    found = True
                    break
        qrs_off.append(iter_sample_end)

    return qrs_on, qrs_off

def get_qrs_area(channel_data, qrs_on, qrs_off, step):
    sum_odd_sample, sum_even_sample = 0, 0

    # Centering the area measurement between the average of the beginning and end values of the QRS complex
    tune = (channel_data[qrs_on] + channel_data[qrs_off])/2

    for iter_odd_sample in range(qrs_on + 1, qrs_off, 2):
        sum_odd_sample = sum_odd_sample + channel_data[iter_odd_sample] - tune

    for iter_even_sample in range(qrs_on, qrs_off, 2):
        if iter_even_sample == qrs_on:
            continue
        sum_even_sample = sum_even_sample + channel_data[iter_even_sample] - tune

    first_sample = channel_data[qrs_on] - tune
    last_sample = channel_data[qrs_off] - tune
    qrs_area = (step/3) * (first_sample + 2*sum_even_sample + 4*sum_odd_sample + last_sample)

    return qrs_area

def get_qrs_angle(qrs_channel_area):
    channel_6_area = qrs_channel_area[5]
    channel_1_area = qrs_channel_area[0]

    qrs_angle = []
    for iter_sample in range(len(channel_1_area)):
        qrs_angle.append(math.degrees(math.atan(channel_6_area[iter_sample] / channel_1_area[iter_sample])))

    return qrs_angle

def get_qrs_median(qrs_intervals):
    on, off = 0, 1
    median_qrs_on = []
    median_qrs_off = []

    iter_qrs = 0
    while iter_qrs < 50:
        current_qrs_on = []
        current_qrs_off = []

        for channel in range(12):
            if iter_qrs < len(qrs_intervals[channel][0]):
                current_qrs_on.append(qrs_intervals[channel][on][iter_qrs])
                current_qrs_off.append(qrs_intervals[channel][off][iter_qrs])

        if len(current_qrs_on) != 0:
            median_qrs_on.append(median_low(current_qrs_on))

        if len(current_qrs_off) != 0:
            median_qrs_off.append(median_high(current_qrs_off))

        iter_qrs += 1

    return median_qrs_on, median_qrs_off

def get_qrs_amplitudes(qrs_peaks, median_qrs_on, data):
    qrs_amplitudes = [[] for i in range(12)]

    for channel in range(12):
        current_difference = [float("{:.4f}".format((data[channel][peak] - data[channel][start]))) for peak, start in zip(qrs_peaks[channel], median_qrs_on)]
        qrs_amplitudes[channel].extend(current_difference)

    return qrs_amplitudes