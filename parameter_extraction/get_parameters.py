import numpy as np

def get_rr_interval(qrs_peaks, f_samp):
    channel2_qrs_peaks = qrs_peaks[1]

    rr_intervals = []
    for iter_peak in range(1, len(channel2_qrs_peaks)):
        rr_intervals.append((channel2_qrs_peaks[iter_peak] - channel2_qrs_peaks[iter_peak-1]) / f_samp)

    return rr_intervals

def get_pr_interval(median_p_on, median_qrs_on, f_samp):
    pr_on = []
    pr_off = []
    pr_interval = []

    for iter_sample in range(len(median_p_on)):
        pr_on.append(median_p_on[iter_sample])
        pr_off.append(median_qrs_on[iter_sample + 1])
        pr_interval.append((pr_off[iter_sample] - pr_on[iter_sample])/f_samp)

    return pr_interval

def get_qt_interval(median_qrs_on, median_t_off, f_samp):
    qt_on = []
    qt_off = []
    qt_interval = []

    for iter_sample in range(len(median_t_off)):
        qt_on.append(median_qrs_on[iter_sample])
        qt_off.append(median_t_off[iter_sample])
        qt_interval.append((qt_off[iter_sample] - qt_on[iter_sample])/f_samp)

    return qt_interval

def get_qtc_interval(qt_interval, beat_interval_mean):
    qtc_interval = []

    for interval in qt_interval:
        try:
            qtc_interval.append(interval / (np.sqrt(beat_interval_mean)))
        except:
            qtc_interval.append(None)

    return qtc_interval

def get_st_deviation(median_qrs_on, median_qrs_off, data):
    st_deviation = [[] for i in range(12)]
    st_on = median_qrs_off[0 : len(median_qrs_off) - 1]
    st_off = median_qrs_on[1 : len(median_qrs_on)]

    for iter_channel in range(12):
        current_amplitude = [float("{:.4f}".format((data[iter_channel][on] - data[iter_channel][off]))) for on, off in zip(st_on, st_off)]
        current_amplitude.append(None)
        st_deviation[iter_channel].extend(current_amplitude)

    return st_deviation

def get_pr_segment(median_p_off, median_qrs_on, f_samp):
    qrs_on = []
    pr_off = []
    pr_segment = []

    for iter_sample in range(len(median_p_off)):
        qrs_on.append(median_p_off[iter_sample])
        pr_off.append(median_qrs_on[iter_sample])
        pr_segment.append((qrs_on[iter_sample] - pr_off[iter_sample])/f_samp)

    return pr_segment

def get_heart_rate(rr_interval):
    heart_rate = []

    for interval in rr_interval:
        heart_rate.append(60/interval)

    return heart_rate