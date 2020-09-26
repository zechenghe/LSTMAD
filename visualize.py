"""
Plot sensor sequences for visualization.
"""

import utils
import numpy as np

train_normal_datasource = 'data/train_normal.npy'
test_normal_datasource = 'data/test_normal.npy'
test_abnormal_datasource = 'data/test_abnormal.npy'

# sensor_idx selects sensor
sensor_idx = 4
seqs = {}
seqs["train_normal"] = np.load(train_normal_datasource)[:, sensor_idx]
seqs["test_normal"] = np.load(test_normal_datasource)[:, sensor_idx]
seqs["test_abnormal"] = np.load(test_abnormal_datasource)[:, sensor_idx]

utils.plot_seq(
    seqs=seqs,
    title="Sensor {sensor_idx}".format(sensor_idx=sensor_idx)
    )
