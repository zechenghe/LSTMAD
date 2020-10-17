import torch
import numpy as np
import scipy.signal

from utils import read_npy_data_single_flle

def load_data_split(split = (0.4, 0.2, 0.4), data_dir = "data/", file_name = 'baseline.npy'):

    assert (len(split) == 3) and (sum(split) == 1.0), "Data split error..."

    normal_data_path = data_dir + file_name
    data = read_npy_data_single_flle(normal_data_path)

    print("Normal data shape ", data.shape)
    assert len(data.shape) == 2, "Normal data should be in shape (TimeFrame, Features)"

    total_length = data.shape[0]
    training_length = int(total_length * split[0])
    ref_length = int(total_length * split[1])
    testing_length = int(total_length * split[2])

    training_normal = data[: training_length, :]
    ref_normal = data[training_length: training_length + ref_length, :]
    testing_normal = data[training_length + ref_length :, :]

    print("Normal data training shape: ", training_normal.shape)
    print("Normal data reference shape: ", ref_normal.shape)
    print("Normal data testing shape: ", testing_normal.shape)

    return np.float32(training_normal), np.float32(ref_normal), np.float32(testing_normal)


def load_data_all(data_dir, file_name):

    abnormal_data_path = data_dir + file_name
    data = read_npy_data_single_flle(abnormal_data_path)

    print("Abormal data shape ", data.shape)
    assert len(data.shape) == 2, "Normal data should be in shape (TimeFrame, Features)"

    return np.float32(data)
