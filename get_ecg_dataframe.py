import os
import wfdb
import time
import numpy as np
import pandas as pd
from natsort import natsorted
from alive_progress import alive_bar
from parameter_extraction.get_ecg_info import create_ecg_info_dict

SAMPLING_FREQUENCY = 500

print('Satisfied requirements...')

start_path = os.getcwd()
path_to_data = os.path.join(start_path, 'physionet.org/files/ludb/1.0.1/data/')

print('Seeking data to be read...')

# Reading dat files
dat_files = []
for file in os.listdir(path_to_data):
    if file.endswith('.dat'):
        dat_files.append(file)
dat_files = natsorted(dat_files)

print('Reading the data...')

'''
Creating a list of electrocardiograms dictionaries containing:
filename, pacient age, pacient sex, electrocardiogram assigned rhythm and the electrocardiogram signal
'''

ecg_data = []
with alive_bar(len(dat_files)) as bar:
    for file in dat_files:
        file_id = file[:-4]

        ecg_info = dict()
        ecg_info['filename'] = 'ecg_extracted_data_' + file_id + '.csv'

        path_to_file = path_to_data + '/' + file_id

        ecg_info['signal'] = []
        for channel in range(12):
            ecg_record = wfdb.rdrecord(path_to_file, channels=[channel])
            ecg_signal = np.concatenate(ecg_record.p_signal)
            ecg_info['signal'].append(ecg_signal)

        ecg_header = wfdb.rdheader(path_to_file).comments
        ecg_assigned_rhythm = ''.join(ecg_header[3].replace('.', '').split()[1:])

        ecg_info['rhythm'] = ecg_assigned_rhythm

        ecg_data.append(ecg_info)
        bar()

print('Generating the data frames for each sample...')

if not os.path.isdir(start_path + '/generated_data'):
    os.mkdir('generated_data')

for index, sample in enumerate(ecg_data):
    with alive_bar(1, title = f'Sample {(index + 1):0{len(str(len(ecg_data)))}d}') as bar:
        ecg_extracted_data = create_ecg_info_dict(sample['signal'], f_samp = SAMPLING_FREQUENCY)
        ecg_extracted_dataframe = pd.DataFrame.from_dict(ecg_extracted_data, orient = 'index').T
        ecg_extracted_dataframe['rhythm'] = sample['rhythm']

        extracted_data_filename = 'generated_data/' + sample['filename']
        ecg_extracted_dataframe.to_csv(extracted_data_filename)
        bar()