"""
    Create multi-attacker data for testing.
"""


import numpy as np
import os

def preprocess(data):
    """
        Preprocess of data.
    """

    acc_mag = np.linalg.norm(data[:, 0:3], axis=-1)
    gyro_mag = acc_norm = np.linalg.norm(data[:, 4:6], axis=-1)
    mag = np.stack([acc_mag, gyro_mag], axis=1)
    return mag

def read_csv(f, delimiter=',', feature_idx=None):
    """
        Read a csv file and return a npy array
        Args:
            f: a file handler
            delimiter: delimiter to split entries
        Returns:
            A np array of shape [T, N_features]
    """

    data = []
    for linenum, line in enumerate(f):
        if linenum != 0:
            data.append(line.split(delimiter))
    data_np = np.array(data, dtype=np.float32)
    if feature_idx is not None:
        data_np = data_np[:, feature_idx]
    #data_np = preprocess(data_np)
    return data_np


data_dir = 'Sensor_Data/task1/'
data_save_dir = 'data/'
user_id = '569'
feature_idx = [0,1,2,3,4,5]

data = {}
with open(os.path.join(data_dir, 'train', '{id}_train.csv'.format(id=user_id)), 'r') as file_handler:
    data['train_normal'] = read_csv(file_handler, feature_idx=feature_idx)

with open(os.path.join(data_dir, 'validation', '{id}_validation.csv'.format(id=user_id)), 'r') as file_handler:
    data['val_and_ref'] = read_csv(file_handler, feature_idx=feature_idx)

with open(os.path.join(data_dir, 'test', '{id}_test.csv'.format(id=user_id)), 'r') as file_handler:
    data['test_normal'] = read_csv(file_handler, feature_idx=feature_idx)

data_attack = []
test_dir = os.path.join(data_dir, 'test')
n_attackers = len(list(os.listdir(test_dir)))-1
n_samples_per_attacker = len(data['test_normal']) / n_attackers
print("n_attackers: ", n_attackers)

for f in os.listdir(test_dir):
    if not f.startswith(user_id):
        print(f)
        with open(os.path.join(test_dir, f), 'r') as file_handler:
            data_per_attacker = read_csv(file_handler, feature_idx=feature_idx)
            l = len(data_per_attacker)
            data_per_attacker = data_per_attacker[l/2: l/2+n_samples_per_attacker, :]
            print(data_per_attacker.shape)
            data_attack.append(data_per_attacker)
data['test_abnormal'] = np.concatenate(data_attack, axis=0)

for k, v in data.items():
    print(k, v.shape)
    np.save(os.path.join(data_save_dir, k), v)
