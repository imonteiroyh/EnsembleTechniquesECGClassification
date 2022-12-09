import warnings
import numpy as np
from parameter_extraction import utils, qrs_complex, t_wave, p_wave, get_parameters, r_peaks_correction

def to_seconds(data, f_samp):
    return float("{:.4f}".format(data / f_samp))

def adjust_data_to_seconds(data, f_samp, removed_begin_sample_size):
    return [to_seconds(i + removed_begin_sample_size, f_samp) if i is not None else None for i in data]

def create_ecg_info_dict(data, f_samp = 240, removed_begin_sample_size = 0):
    acl = []
    ecg_extracted_data = {}

    for iter in range(removed_begin_sample_size):
      data = np.delete(data, 0, axis = 1)
      data = np.delete(data, -1, axis = 1)

    # Baseline filter application
    for iter_channel in range(12):
        data[iter_channel] = utils.baseline_filter(data[iter_channel], f_samp, 8)

    floating_window_size = round(0.04 * f_samp)
    delay = floating_window_size

    # QRS complex
    qrs_peaks = qrs_complex.get_qrs_peak(data, f_samp)

    wavelet_scale = 3
    qrs_intervals = []
    for iter_channel in range(12):
        wavelet_transformed_data = utils.wavelet_transform(data[iter_channel], wavelet_scale)
        current_acl = utils.get_acl(wavelet_transformed_data, floating_window_size)
        acl.append(current_acl)
        qrs_on, qrs_off = qrs_complex.get_qrs_interval(qrs_peaks, current_acl, iter_channel, f_samp, delay)
        qrs_intervals.append((qrs_on, qrs_off))

    median_qrs_on, median_qrs_off = qrs_complex.get_qrs_median(qrs_intervals)
    ecg_extracted_data['qrs_start'] = adjust_data_to_seconds(median_qrs_on, f_samp, removed_begin_sample_size)
    ecg_extracted_data['qrs_end'] = adjust_data_to_seconds(median_qrs_off, f_samp, removed_begin_sample_size)
    ecg_extracted_data['qrs_interval'] = [to_seconds(end - start, f_samp) for end, start in zip(median_qrs_off, median_qrs_on)]

    qrs_amplitudes = qrs_complex.get_qrs_amplitudes(qrs_peaks, median_qrs_on, data)
    for iter_channel in range(12):
        ecg_extracted_data[f'qrs_amplitude_channel_{iter_channel}'] = qrs_amplitudes[iter_channel]

    qrs_area = []
    for iter_channel in range(12):
        qrs_area.append([])
        for iter_qrs in range(len(median_qrs_on)):
            current_qrs_area = qrs_complex.get_qrs_area(data[iter_channel], median_qrs_on[iter_qrs], median_qrs_off[iter_qrs], 1 / f_samp)
            qrs_area[iter_channel].append(current_qrs_area)
    for iter_channel in range(12):
        ecg_extracted_data[f'qrs_area_channel_{iter_channel}'] = [float("{:.4f}".format(i)) for i in qrs_area[iter_channel]]

    qrs_angle = qrs_complex.get_qrs_angle(qrs_area)
    ecg_extracted_data['qrs_angle'] = [float("{:.4f}".format(i)) for i in qrs_angle]

    # T wave
    search_t_on, search_t_off = t_wave.search_t_interval(median_qrs_on, median_qrs_off, delay)
    t_peaks = t_wave.get_t_peak(data, acl, search_t_on, search_t_off, delay)
    for iter_channel in range(12):
        ecg_extracted_data[f't_peaks_channel_{iter_channel}'] = adjust_data_to_seconds(t_peaks[iter_channel], f_samp, removed_begin_sample_size)

    t_intervals = []
    for iter_channel in range(12):
        current_acl = acl[iter_channel]
        current_qrs_off = qrs_intervals[iter_channel][1]
        t_on, t_off = t_wave.get_t_interval(t_peaks, current_qrs_off, current_acl, iter_channel, f_samp, delay)
        t_intervals.append((t_on, t_off))

    median_t_on, median_t_off = t_wave.get_t_median(t_intervals)
    ecg_extracted_data['t_start'] = adjust_data_to_seconds(median_t_on, f_samp, removed_begin_sample_size)
    ecg_extracted_data['t_end'] = adjust_data_to_seconds(median_t_off, f_samp, removed_begin_sample_size)
    ecg_extracted_data['t_interval'] = [to_seconds(end - start, f_samp) for end, start in zip(median_t_off, median_t_on)]

    t_amplitudes = t_wave.get_t_amplitudes(t_peaks, median_t_off, data)
    for iter_channel in range(12):
        ecg_extracted_data[f't_amplitude_channel_{iter_channel}'] = t_amplitudes[iter_channel]

    t_area = []
    for iter_channel in range(12):
        t_area.append([])
        for iter_t in range(len(median_t_on)):
            current_t_area = t_wave.get_t_area(data[iter_channel], median_t_on[iter_t], median_t_off[iter_t], 1/f_samp)
            t_area[iter_channel].append(current_t_area)
        ecg_extracted_data[f't_area_channel_{iter_channel}'] = [float("{:.4f}".format(i)) for i in t_area[iter_channel]]

    t_angle = t_wave.get_t_angle(t_area)
    ecg_extracted_data['t_angle'] = [float("{:.4f}".format(i)) for i in t_angle]

    # P wave
    search_p_on, search_p_off = p_wave.search_p_interval(median_qrs_on, median_qrs_off, delay)
    p_peaks = p_wave.get_p_peak(data, acl, search_p_on, search_p_off, delay)
    for iter_channel in range(12):
        ecg_extracted_data[f'p_peaks_channel_{iter_channel}'] = adjust_data_to_seconds(p_peaks[iter_channel], f_samp, removed_begin_sample_size)

    p_intervals = []
    for iter_channel in range(12):
        current_acl = acl[iter_channel]
        current_qrs_on = qrs_intervals[iter_channel][0]
        p_on, p_off = p_wave.get_p_interval(p_peaks, current_qrs_on, current_acl, iter_channel, f_samp, delay)
        p_intervals.append((p_on, p_off))

    median_p_on, median_p_off = p_wave.get_p_median(p_intervals)
    ecg_extracted_data['p_start'] = adjust_data_to_seconds([np.nan] + median_p_on, f_samp, removed_begin_sample_size)
    ecg_extracted_data['p_end'] = adjust_data_to_seconds([np.nan] + median_p_off, f_samp, removed_begin_sample_size)
    ecg_extracted_data['p_interval'] = [to_seconds(end - start, f_samp) for end, start in zip(median_p_off, median_p_on)]

    p_amplitudes = p_wave.get_p_amplitudes(p_peaks, median_qrs_on, data)
    for iter_channel in range(12):
        ecg_extracted_data[f'p_amplitude_channel_{iter_channel}'] = p_amplitudes[iter_channel]

    p_area = []
    for iter_channel in range(12):
        p_area.append([])
        for iter_p in range(len(median_p_on)):
            current_p_area = p_wave.get_p_area(data[iter_channel], median_p_on[iter_p], median_p_off[iter_p], 1/f_samp)
            p_area[iter_channel].append(current_p_area)
        ecg_extracted_data[f'p_area_channel_{iter_channel}'] = [float("{:.4f}".format(i)) for i in p_area[iter_channel]]

    p_angle = p_wave.get_p_angle(p_area)
    ecg_extracted_data['p_angle'] = [float("{:.4f}".format(i)) for i in p_angle]

    # R peaks correction by the mean beat method
    qrs_peaks = r_peaks_correction.get_corrected_peaks(data, qrs_peaks, median_p_on, median_t_off)
    for iter_channel in range(12):
        ecg_extracted_data[f'qrs_peaks_channel_{iter_channel}'] = adjust_data_to_seconds(qrs_peaks[iter_channel], f_samp, removed_begin_sample_size)

    # Extracted parameters from QRS complex, T wave and P wave info
    rr_interval = get_parameters.get_rr_interval(qrs_peaks, f_samp)
    ecg_extracted_data['rr_interval'] = [float("{:.4f}".format(i)) for i in rr_interval]

    pr_interval = get_parameters.get_pr_interval(median_p_on, median_qrs_on, f_samp)
    ecg_extracted_data['pr_interval'] = [float("{:.4f}".format(i)) for i in pr_interval]

    qt_interval = get_parameters.get_qt_interval(median_qrs_on, median_t_off, f_samp)
    ecg_extracted_data['qt_interval'] = [float("{:.4f}".format(i)) for i in qt_interval]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            beat_intervals_mean = np.mean(rr_interval)
            qtc_interval = get_parameters.get_qtc_interval(qt_interval, beat_intervals_mean)
            ecg_extracted_data['qtc_interval'] = [float("{:.4f}".format(i)) for i in qtc_interval]
    except:
        beat_intervals_mean = None
        ecg_extracted_data['qtc_interval'] = []

    pr_segment = get_parameters.get_pr_segment(median_p_off, median_qrs_on, f_samp)
    ecg_extracted_data['pr_segment'] = [float("{:.4f}".format(i)) for i in pr_segment]

    heart_rate = get_parameters.get_heart_rate(rr_interval)
    ecg_extracted_data['heart_rate'] = [float("{:.4f}".format(i)) for i in heart_rate]

    st_deviation = get_parameters.get_st_deviation(median_qrs_on, median_qrs_off, data)
    for iter_channel in range(12):
        ecg_extracted_data[f'st_deviation_channel_{iter_channel}'] = st_deviation[iter_channel]

    return ecg_extracted_data