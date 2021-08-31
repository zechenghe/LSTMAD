### LSTM anormaly detection

Implementation of LSTM + KS test for anomaly detection in the following paper:

Zecheng He, Aswin Raghavan, Guangyuan Hu, Sek Chai, and Ruby Lee. "[Power-grid Controller Anomaly Detection with Enhanced Temporal Deep Learning](https://ieeexplore.ieee.org/abstract/document/8887367?casa_token=IAE4kv3Nc_0AAAAA:udZTvOo60xKOlqYal80eOaPdByUNfP03raQlESzB0Y2ub1s8qxEbn_9KmQVmF_ttQ2NzjOZ2k-o)". IEEE International Conference On Trust, Security And Privacy In Computing And Communications (TrustCom), 2019.

#### Sample data

The original data of power-grid controller is under confidentiality restriction. Here we provide a sample data of [smartphone impostor detection](https://arxiv.org/pdf/2103.06453.pdf). The data format is the same.

The sample data (smartphone sensor data) are in `data/`: `train_normal.npy`, `val_and_ref.npy`, `test_normal.npy`, `test_abnormal.npy`.


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

| Hyper-parameters | Description | Default Value | Recommended Tuning Range
|:-:|:-:|:-:|:-:|
| Nhidden | The number of hidden nodes in the LSTM cell | 64 | 16, 32, 64, ..., 1024 |
| Nbatches | Number of batches for training the model | 100 | 50, 100, 200, 500, 1000 |
| BatchSize | Batch size for training the model | 16 | 4, 8, 16, 32, 64, 128 |
| ChunkSize | Length of a chunk for training | 500 | 100, 200, 500, 1000, 2000, 5000 |
| SubseqLen | Length of randomly selected sequences for training | 5000 | 50, 100, 200, 500, 1000, 2000, 5000 |
| LearningRate | Learning rate for training | 1e-2 | 1e-1, 1e-2, 1e-3, 1e-4 |
| RED_collection_len | The number of prediction errors accumulated as a RED point | 1 | 1, 10, 20, 50 |
| RED_points | The number of points to form a RED distribution | 100 | 10, 20, 50, 100, 200, 500, 1000 |
| Pvalue_th | p-value threshold in KS-test to determine abnormal | 0.05 | Tune only after you have already got a good ROC. Find threshold at EER is automatic. |

#### Reference
You are encouraged to cite the following paper.

```
@inproceedings{he2019power,
  title={Power-grid controller anomaly detection with enhanced temporal deep learning},
  author={He, Zecheng and Raghavan, Aswin and Hu, Guangyuan and Chai, Sek and Lee, Ruby},
  booktitle={IEEE International Conference On Trust, Security And Privacy In Computing And Communications (TrustCom)},
  year={2019}
}
```
