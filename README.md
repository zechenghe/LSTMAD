Training and testing are sequenctial data in the form of [TimeFrame, NFeatures]

Create a data directory `data/`
```shell
mkdir data
```

Put the collected sensor data (train_normal, test_normal, test_abnormal) under directory `data/` and run preprocessing:
```shell
python preprocess_data_ELE472.py
```
It generates `.npy` file from `.csv` for training

Modeing training:
```shell
python LSTMAD.py --training --normal_data_name_train test_abnormal.npy
```

Modeing testing:
```shell
python LSTMAD.py --testing --normal_data_name_test test_normal.npy --abnormal_data_name test_abnormal.npy --Pvalue_th 1e-3
```
