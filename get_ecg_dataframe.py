import os
import wfdb
import numpy as np
import pandas as pd
from natsort import natsorted
from get_ecg_info import create_ecg_info_dict

SAMPLING_FREQUENCY = 500

start_path = os.getcwd()
path_to_data = os.path.join(start_path, 'physionet.org/files/ludb/1.0.1/data/')

# Reading dat files
dat_files = []
for file in os.listdir(path_to_data):
    if file.endswith('.dat'):
        dat_files.append(file)
dat_files = natsorted(dat_files)


dat_files = dat_files[:5]

'''
Creating a list of electrocardiograms dictionaries containing:
filename, pacient age, pacient sex, electrocardiogram assigned rhythm and the electrocardiogram signal
'''
ecg_data = []
for file in dat_files:
    file_id = file[:-4]

    ecg_info = dict()
    ecg_info['filename'] = file

    path_to_file = path_to_data + '/' + file_id

    ecg_info['signal'] = []
    for channel in range(12):
        ecg_record = wfdb.rdrecord(path_to_file, channels=[channel])
        ecg_signal = np.concatenate(ecg_record.p_signal)
        ecg_info['signal'].append(ecg_signal)

    ecg_header = wfdb.rdheader(path_to_file).comments
    pacient_age = ecg_header[0].split()[1]
    pacient_sex = ecg_header[1].split()[1]
    ecg_assigned_rhythm = ''.join(ecg_header[3].replace('.', '').split()[1:])

    ecg_info['age'] = pacient_age
    ecg_info['sex'] = pacient_sex
    ecg_info['rhythm'] = ecg_assigned_rhythm

    ecg_data.append(ecg_info)

for sample in ecg_data:
    ecg_extracted_data = create_ecg_info_dict(sample['signal'], f_samp = SAMPLING_FREQUENCY)
    ecg_extracted_dataframe = pd.DataFrame.from_dict(ecg_extracted_data, orient = 'index').T

    print(ecg_extracted_dataframe)