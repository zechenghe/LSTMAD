"""
Plot sensor sequences for visualization.
"""

import utils
import numpy as np
import matplotlib.pyplot as plt

train_normal_datasource = 'data/train_normal.npy'
val_ref_normal_datasource = 'data/val_and_ref.npy'
test_normal_datasource = 'data/test_normal.npy'
test_abnormal_datasource = 'data/test_abnormal.npy'

# sensor_idx selects sensor
sensor_idx = 0
start = 0
end = 8000
utils.plot_seq(
    seqs={
        "train_normal": np.load(train_normal_datasource)[:, sensor_idx],
        "val_ref_normal": np.load(val_ref_normal_datasource)[:, sensor_idx]
    },
    start=start,
    T=end,
    title="Sensor {sensor_idx}".format(sensor_idx=sensor_idx)
    )

utils.plot_seq(
    seqs={
        "train_normal": np.load(train_normal_datasource)[:, sensor_idx],
        "test_normal": np.load(test_normal_datasource)[:, sensor_idx]
    },
    start=start,
    T=end,
    title="Sensor {sensor_idx}".format(sensor_idx=sensor_idx)
    )

utils.plot_seq(
    seqs={
        "train_normal": np.load(train_normal_datasource)[:, sensor_idx],
        "test_abnormal": np.load(test_abnormal_datasource)[:, sensor_idx]
    },
    start=start,
    T=end,
    title="Sensor {sensor_idx}".format(sensor_idx=sensor_idx)
    )

plt.show()
