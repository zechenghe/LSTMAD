import matplotlib.pyplot as plt
import numpy as np

train_normal_datasource = 'data/train_normal.npy'
test_normal_datasource = 'data/test_normal.npy'
test_abnormal_datasource = 'data/test_abnormal.npy'

feature_idx = 7
train_normal = np.load(train_normal_datasource)[:, feature_idx]
test_normal = np.load(test_normal_datasource)[:, feature_idx]
test_abnormal = np.load(test_abnormal_datasource)[:, feature_idx]

plt.figure()
t = np.arange(0, 500.0, 1.0)
train_normal_plot, = plt.plot(t, train_normal[:500])
train_normal_plot.set_label("Train normal")
test_normal_plot, = plt.plot(t, test_normal[:500])
test_normal_plot.set_label("Test normal")
test_abnormal, = plt.plot(t, test_abnormal[:500])
test_abnormal.set_label("Test abnormal")
plt.xlabel('Time frame')
plt.ylabel('Sensor readings')
plt.title('Sensor readings')
plt.legend(loc="lower right")
plt.show()
