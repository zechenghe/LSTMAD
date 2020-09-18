'''
Covert csv data to npy arrays of shape [TimeFrame, Features]
'''

import numpy as np
import os

data_dir = 'data/'
for f in os.listdir(data_dir):
    file_name_split = f.split('.')
    if file_name_split[-1] == 'csv':
        with open(data_dir+f, 'r') as file_handler:
            data = []
            for line in file_handler:
                # Remove sensor names
                data.append([float(x) for x in line.split(',')[1:]])
            # Convert data to np array and transpose to [TimeFrame, Features]
            # Use Accelerometer, Gyroscope and Magnetometer
            data_np = np.transpose(np.array(data))
        np.save(data_dir + file_name_split[0] + '.npy', data_np)
        data_load = np.load(data_dir + file_name_split[0] + '.npy')
        print file_name_split[0], data_load.shape
