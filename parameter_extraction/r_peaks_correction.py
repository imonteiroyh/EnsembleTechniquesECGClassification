import numpy as np
from scipy.signal import correlate
from statistics import median_low

def get_corrected_peaks(data, qrs_peaks, median_p_on, median_t_off):
    corrected_peak_positions = []

    for iter_channel in range(12):
        try:
            current_qrs_peaks = qrs_peaks[iter_channel][1 : len(qrs_peaks[iter_channel]) - 1]
            current_channel_beats = [[] for i in range(len(current_qrs_peaks))]

            left_distance = []
            for iter_beat in range(len(current_qrs_peaks)):
                left_distance.append(current_qrs_peaks[iter_beat] - median_p_on[iter_beat])

            try:
                greatest_left_distance = max(left_distance)
            except:
                greatest_left_distance = 0

            for iter_beat in range(len(current_qrs_peaks)):
                current_channel_beats[iter_beat].extend([0 for i in range(greatest_left_distance - left_distance[iter_beat])])
                current_channel_beats[iter_beat].extend(data[iter_channel][median_p_on[iter_beat] : current_qrs_peaks[iter_beat]])

            right_distance = []
            for iter_beat in range(len(current_qrs_peaks)):
                right_distance.append(median_t_off[iter_beat + 1] - current_qrs_peaks[iter_beat])

            try:
                greatest_right_distance = max(right_distance)
            except:
                greatest_right_distance = 0

            for iter_beat in range(len(current_qrs_peaks)):
                current_channel_beats[iter_beat].extend(data[iter_channel][current_qrs_peaks[iter_beat] : median_t_off[iter_beat + 1] + 1])
                current_channel_beats[iter_beat].extend([0 for i in range(greatest_right_distance - right_distance[iter_beat])])

            total_beat = [0 for i in range(len(current_channel_beats[0]))]
            for iter_beat in range(len(current_qrs_peaks)):
                total_beat = [total + current for total, current in zip(total_beat, current_channel_beats[iter_beat])]

            current_mean_beat_data = [sample / len(current_qrs_peaks) for sample in total_beat]

            peak_positions = []
            for iter_beat in range(len(current_qrs_peaks)):
                correlation = correlate(current_channel_beats[iter_beat], current_mean_beat_data)

                current_peak_position = np.where(correlation == np.amax(correlation))[0][0]
                peak_positions.append(current_peak_position)

            median_peak_position = median_low(peak_positions)
            for iter_beat in range(len(current_qrs_peaks)):
                current_difference = median_peak_position - peak_positions[iter_beat]
                if current_difference > 0:
                    while (current_difference > 0):
                        current_channel_beats[iter_beat].insert(0, 0)
                        current_channel_beats[iter_beat].pop()
                        current_difference = current_difference - 1
                elif current_difference < 0:
                    while (current_difference < 0):
                        current_channel_beats[iter_beat].insert(-1, 0)
                        current_channel_beats[iter_beat].pop(0)
                        current_difference = current_difference + 1

            corrected_peak_channel = []
            corrected_peak_channel.append(qrs_peaks[iter_channel][0])

            for iter_beat in range(len(current_qrs_peaks)):
                current_peak = current_qrs_peaks[iter_beat]
                current_difference = peak_positions[iter_beat] - median_low(peak_positions)
                corrected_peak_channel.append(current_peak - current_difference)

            corrected_peak_channel.append(qrs_peaks[iter_channel][-1])
            corrected_peak_positions.append(corrected_peak_channel)
        except:
            corrected_peak_positions.append(qrs_peaks[iter_channel])

    return corrected_peak_positions