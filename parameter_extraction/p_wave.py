import math
import numpy as np
from statistics import median_low, median_high

def search_p_interval(median_qrs_on, median_qrs_off, delay):
    search_p_on = []
    search_p_on.extend(median_qrs_off)
    search_p_on.pop()

    search_p_off = []
    search_p_off.extend(median_qrs_on)
    search_p_off.pop(0)

    for iter in range(len(search_p_off)):
        search_p_off[iter] = search_p_off[iter] - delay
        search_p_on[iter] = int(search_p_on[iter] + (search_p_off[iter] - search_p_on[iter]) / 2)

    return search_p_on, search_p_off

def get_p_peak(data, acl, p_on, p_off, delay):
    p_peaks = []

    for iter_channel in range(12):
        current_channel_p_peaks = []
        current_acl = acl[iter_channel]

        for iter in range(len(p_on)):
            search_area = current_acl[p_on[iter] : p_off[iter]]
            try:
                current_p_peak = np.where(search_area == np.amax(search_area))[0][0]
            except:
                continue
            current_channel_p_peaks.append(current_p_peak + p_on[iter])

        current_channel_real_p_peaks = []
        current_channel_data = np.copy(data[iter_channel])
        for iter_peak in range(len(current_channel_p_peaks)):
            current_peak = current_channel_p_peaks[iter_peak]

            if (current_peak - delay) >= 0:
                if current_channel_data[current_peak] >= 0:
                    current_channel_real_p_peaks.append(current_peak - delay + np.argmax(current_channel_data[current_peak - delay : current_peak + delay]))
                else:
                    current_channel_real_p_peaks.append(current_peak - delay + np.argmin(current_channel_data[current_peak - delay : current_peak + delay]))
            else:
                if current_channel_data[current_peak] >= 0:
                    current_channel_real_p_peaks.append(np.argmax(current_channel_data[0 : current_peak + delay]))
                else:
                    current_channel_real_p_peaks.append(np.argmin(current_channel_data[0 : current_peak + delay]))

        p_peaks.append(current_channel_real_p_peaks)

    return p_peaks

def get_p_interval(p_peaks, qrs_on, acl, channel, f_samp, delay):
    p_on = []
    p_off = []

    for iter_peak in range(len(p_peaks[channel])):

        iter_sample_on = p_peaks[channel][iter_peak]
        if (iter_sample_on - round(0.12*f_samp) >= 0):
            window = acl[(iter_sample_on - round(0.12*f_samp)) : (iter_sample_on + round(0.12*f_samp))]
        else:
            window = acl[0 : (iter_sample_on + round(0.12*f_samp))]

        found = False
        while not(found):
            if acl[iter_sample_on] < 1.2*acl[iter_sample_on - 1] and acl[iter_sample_on] < 1.2*acl[iter_sample_on + 1] and p_peaks[channel][iter_peak] - (iter_sample_on + delay) >= 0.04*f_samp and acl[iter_sample_on] < 0.7*max(window):
                found = True
                break
            else:
                iter_sample_on = iter_sample_on - 1

            if iter_sample_on <= 0:
                iter_sample_on = - delay
                found = True
                break
        p_on.append(iter_sample_on + delay)

        iter_sample_end = p_peaks[channel][iter_peak]
        min_value = acl[iter_sample_end]
        min_position = iter_sample_end
        found = False
        while not(found):
            if iter_sample_end >= qrs_on[iter_peak + 1]:
                found = True
                break
            if acl[iter_sample_end] < 1.2*acl[iter_sample_end - 1] and acl[iter_sample_end] < 1.2*acl[iter_sample_end + 1] and iter_sample_end - p_peaks[channel][iter_peak] >= 0.04*f_samp and acl[iter_sample_end] < 0.7*max(window):
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
        p_off.append(iter_sample_end)

    return p_on, p_off

def get_p_median(p_intervals):
    on, off = 0, 1
    median_p_on = []
    median_p_off = []

    for iter_p in range(len(p_intervals[0][0])):
        current_p_on = []
        current_p_off = []

        for channel in range(12):
            current_p_on.append(p_intervals[channel][on][iter_p])
            current_p_off.append(p_intervals[channel][off][iter_p])

        median_p_on.append(median_low(current_p_on))
        median_p_off.append(median_high(current_p_off))

    return median_p_on, median_p_off

def get_p_area(channel_data, p_on, p_off, step):
    sum_odd_sample, sum_even_sample = 0, 0

    # Centering the area measurement between the average of the beginning and end values of the P wave
    tune = (channel_data[p_on] + channel_data[p_off])/2

    for iter_odd_sample in range(p_on + 1, p_off, 2):
        sum_odd_sample = sum_odd_sample + channel_data[iter_odd_sample] - tune

    for iter_even_sample in range(p_on, p_off, 2):
        if iter_even_sample == p_on:
            continue
        sum_even_sample = sum_even_sample + channel_data[iter_even_sample] - tune

    first_sample = channel_data[p_on] - tune
    last_sample = channel_data[p_off] - tune
    p_area = (step/3) * (first_sample + 2*sum_even_sample + 4*sum_odd_sample + last_sample)

    return p_area

def get_p_angle(p_channel_area):
    channel_6_area = p_channel_area[5]
    channel_1_area = p_channel_area[0]

    p_angle = []
    for iter_sample in range(len(channel_1_area)):
        p_angle.append(math.degrees(math.atan(channel_6_area[iter_sample] / channel_1_area[iter_sample])))

    return p_angle

def get_p_amplitudes(p_peaks, median_qrs_on, data):
    p_amplitudes = [[] for i in range(12)]
    median_qrs_on = median_qrs_on[1:len(median_qrs_on)]
    for channel in range(12):
        current_difference = [float("{:.4f}".format((data[channel][peak] - data[channel][start]))) for peak, start in zip(p_peaks[channel], median_qrs_on)]
        current_difference.append(None)
        p_amplitudes[channel].extend(current_difference)

    return p_amplitudes