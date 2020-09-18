#### Preprocess data

Create a data directory `data/`
```shell
mkdir data
```

Put the collected sensor data in HW1 (train_normal, test_normal, test_abnormal) under directory `data/` and run preprocessing:
```shell
python preprocess_data_ELE472.py
```
It generates `.npy` files from `.csv` for training

#### Train an LSTM anomaly detector

Modeing training:
```shell
python LSTMAD.py --training --normal_data_name_train train_normal.npy
```

#### Evaluate the anomaly detector

Modeing testing:
```shell
python LSTMAD.py --testing --normal_data_name_test test_normal.npy --abnormal_data_name test_abnormal.npy --Pvalue_th 1e-3
```

#### Train and test with different hyper-parameters

| Hyper-parameters | Description |
|:-:|:-:|
| Nhidden | The number of hidden nodes in the LSTM cell |
| Nbatches | Number of batches for training the model |
| BatchSize | Batch size for training the model |
| ChunkSize | Length of a chunk for training |
| SubseqLen | Length of randomly selected sequences for training |
| LearningRate | Learning rate for training |
| RED_collection_len | The number of prediction errors accumulated as a RED point |
| RED_points | The number of points to form a RED distribution |
| Pvalue_th | p-value threshold in KS-test to determine abnormal |


#### Visualize sensor readings
```shell
python visualize.py
```
