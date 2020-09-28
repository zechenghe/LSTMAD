### LSTM anormaly detection

Implementation of LSTM + KS test anomaly detection

#### Preprocess data

Create a data directory `data/`
```shell
mkdir data
```

Put the collected sensor data in HW1 (train_normal, val_and_ref, test_normal, test_abnormal) under directory `data/`. Better to rename the corresponding data to `train_normal.csv`, `val_and_ref.csv`, `test_normal.csv`, `test_abnormal.csv` to match the default file names in the training and evaluation scripts.

Run preprocess script:
```shell
python preprocess_data_ELE472.py
```
It generates `.npy` files from `.csv` for training. It uses accelerometer and gyroscope magnitude. Feel free to add more sensors.

#### Visualize sensor readings
```shell
python visualize.py
```

#### Train an LSTM anomaly detector

Modeing training (with default parameters):
```shell
python LSTMAD.py --training
```

#### Evaluate the anomaly detector

Modeing evaluation:
```shell
python LSTMAD.py --testing --Pvalue_th 5e-2
```

`--debug` option is useful for printing and plotting reconstruction errors and other useful information to help debug and select the hyper-parameters.

First, tune training parameters to make sure the ROC-AUC > 0.5 before you change the threshold `Pvalue_th`. The top-left region is the best on a ROC plot.

Each threshold `Pvalue_th` corresponding to a point on the ROC curve. Change `Pvalue_th` to move on the ROC curve.

#### Train and test with different hyper-parameters

For example, train with Nhidden=32, BatchSize=32 and ChunkSize=1000:

```shell
python LSTMAD.py --training --Nhidden 32 --BatchSize 32 --ChunkSize 1000
```

| Hyper-parameters | Description | Default Value |
|:-:|:-:|:-:|
| Nhidden | The number of hidden nodes in the LSTM cell | 64 |
| Nbatches | Number of batches for training the model | 100 |
| BatchSize | Batch size for training the model | 16 |
| ChunkSize | Length of a chunk for training | 500 |
| SubseqLen | Length of randomly selected sequences for training | 5000 |
| LearningRate | Learning rate for training | 1e-2 |
| RED_collection_len | The number of prediction errors accumulated as a RED point | 1 |
| RED_points | The number of points to form a RED distribution | 100 |
| Pvalue_th | p-value threshold in KS-test to determine abnormal | 0.05 |
