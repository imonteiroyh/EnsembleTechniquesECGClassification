import math
import numpy as np
from statistics import median_low, median_high

def search_t_interval(median_qrs_on, median_qrs_off, delay):
    search_t_on = []
    search_t_on.extend(median_qrs_off)
    search_t_on.pop()

    search_t_off = []
    search_t_off.extend(median_qrs_on)
    search_t_off.pop(0)

    for iter in range(len(search_t_off)):
        search_t_off[iter] = search_t_off[iter] - delay
        search_t_off[iter] = int(search_t_on[iter] + (search_t_off[iter] - search_t_on[iter]) / 2)

    return search_t_on, search_t_off

def get_t_peak(data, acl, t_on, t_off, delay):
    t_peaks = []

    for iter_channel in range(12):
        current_channel_t_peaks = []
        current_acl = acl[iter_channel]

        for iter in range(len(t_on)):
            search_area = current_acl[t_on[iter] : t_off[iter]]
            try:
                current_t_peak = np.where(search_area == np.amax(search_area))[0][0]
            except:
                continue
            current_channel_t_peaks.append(current_t_peak + t_on[iter])

        current_channel_real_t_peaks = []
        current_channel_data = np.copy(data[iter_channel])
        for iter_peak in range(len(current_channel_t_peaks)):
            current_peak = current_channel_t_peaks[iter_peak]

            if (current_peak - delay) >= 0:
                if current_channel_data[current_peak] >= 0:
                    current_channel_real_t_peaks.append(current_peak - delay + np.argmax(current_channel_data[current_peak - delay : current_peak + delay]))
                else:
                    current_channel_real_t_peaks.append(current_peak - delay + np.argmin(current_channel_data[current_peak - delay : current_peak + delay]))
            else:
                if current_channel_data[current_peak] >= 0:
                    current_channel_real_t_peaks.append(np.argmax(current_channel_data[0 : current_peak + delay]))
                else:
                    current_channel_real_t_peaks.append(np.argmin(current_channel_data[0 : current_peak + delay]))

        t_peaks.append(current_channel_real_t_peaks)

    return t_peaks

def get_t_interval(t_peaks, qrs_off, acl, channel, f_samp, delay):
    t_on = []
    t_off = []

    for iter_peak in range(len(t_peaks[channel])):

        iter_sample_on = t_peaks[channel][iter_peak]
        if (iter_sample_on - round(0.12*f_samp) >= 0):
            window = acl[(iter_sample_on - round(0.12*f_samp)) : (iter_sample_on + round(0.12*f_samp))]
        else:
            window = acl[0 : (iter_sample_on + round(0.12*f_samp))]

        found = False
        while not(found):
            if iter_sample_on <= qrs_off[iter_peak]:
                found = True
                break
            if acl[iter_sample_on] < 1.1*acl[iter_sample_on - 1] and acl[iter_sample_on] < 1.1*acl[iter_sample_on + 1] and t_peaks[channel][iter_peak] - (iter_sample_on + delay) >= 0.1*f_samp and acl[iter_sample_on] < 0.7*max(window):
                found = True
                break
            else:
                iter_sample_on = iter_sample_on - 1
        t_on.append(iter_sample_on)

        iter_sample_end = t_peaks[channel][iter_peak]
        min_value = acl[iter_sample_end]
        min_position = iter_sample_end
        found = False
        while not(found):
            if acl[iter_sample_end] < 1.1*acl[iter_sample_end - 1] and acl[iter_sample_end] < 1.1*acl[iter_sample_end + 1] and iter_sample_end - t_peaks[channel][iter_peak] >= 0.1*f_samp and acl[iter_sample_end] < 0.7*max(window):
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
        t_off.append(iter_sample_end)

    return t_on, t_off

def get_t_median(t_intervals):
    on, off = 0, 1
    median_t_on = []
    median_t_off = []

    for iter_t in range(len(t_intervals[0][0])):
        current_t_on = []
        current_t_off = []

        for channel in range(12):
            current_t_on.append(t_intervals[channel][on][iter_t])
            current_t_off.append(t_intervals[channel][off][iter_t])

        median_t_on.append(median_low(current_t_on))
        median_t_off.append(median_high(current_t_off))

    return median_t_on, median_t_off

def get_t_area(channel_data, t_on, t_off, step):
    sum_odd_sample, sum_even_sample = 0, 0

    # Centering the area measurement between the average of the beginning and end values of the T wave
    tune = (channel_data[t_on] + channel_data[t_off])/2

    for iter_odd_sample in range(t_on + 1, t_off, 2):
        sum_odd_sample = sum_odd_sample + channel_data[iter_odd_sample] - tune

    for iter_even_sample in range(t_on, t_off, 2):
        if iter_even_sample == t_on:
            continue
        sum_even_sample = sum_even_sample + channel_data[iter_even_sample] - tune

    first_sample = channel_data[t_on] - tune
    last_sample = channel_data[t_off] - tune
    t_area = (step/3) * (first_sample + 2*sum_even_sample + 4*sum_odd_sample + last_sample)

    return t_area

def get_t_angle(t_channel_area):
    channel_6_area = t_channel_area[5]
    channel_1_area = t_channel_area[0]

    t_angle = []
    for iter_sample in range(len(channel_1_area)):
        t_angle.append(math.degrees(math.atan(channel_6_area[iter_sample] / channel_1_area[iter_sample])))

    return t_angle

def get_t_amplitudes(t_peaks, median_t_off, data):
    t_amplitudes = [[] for i in range(12)]

    for channel in range(12):
        current_difference = [float("{:.4f}".format((data[channel][peak] - data[channel][end]))) for peak, end in zip(t_peaks[channel], median_t_off)]
        t_amplitudes[channel].extend(current_difference)

    return t_amplitudes