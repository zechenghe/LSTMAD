import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import warnings
# Ignore warnings due to pytorch save models
# https://github.com/pytorch/pytorch/issues/27972
warnings.filterwarnings("ignore", "Couldn't retrieve source code")

import time
import math
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

import utils
import SeqGenerator
import detector
import loaddata

def eval_detector(args):

    load_model_dir = args.load_model_dir
    load_model_name = args.load_model_name
    normal_data_dir = args.normal_data_dir
    normal_data_name_train = args.normal_data_name_train
    normal_data_name_test = args.normal_data_name_test
    abnormal_data_dir = args.abnormal_data_dir
    abnormal_data_name = args.abnormal_data_name
    Pvalue_th = args.Pvalue_th
    val_and_ref_name = args.normal_data_name_val_and_ref

    gpu = args.gpu

    AnomalyDetector = torch.load(load_model_dir + load_model_name)
    AnomalyDetector.eval()
    AnomalyDetector.th = Pvalue_th

    if args.dummydata:
        _, _, testing_normal_data = loaddata.load_normal_dummydata()
    else:
        if args.debug:
            _, training_normal_data, _, = (
                loaddata.load_data_split(
                    data_dir = normal_data_dir,
                    file_name = normal_data_name_train,
                    split = (0.1, 0.8, 0.1)
                    )
                    )

            _, ref_normal_data, val_normal_data = loaddata.load_data_split(
                data_dir = normal_data_dir,
                file_name = val_and_ref_name,
                split = (0.1, 0.45, 0.45)
                )

            training_normal_data = torch.tensor(
                AnomalyDetector.normalize(training_normal_data))
            val_normal_data = torch.tensor(
                AnomalyDetector.normalize(val_normal_data))
            ref_normal_data = torch.tensor(
                AnomalyDetector.normalize(ref_normal_data))


        testing_normal_data = loaddata.load_data_all(
            data_dir = normal_data_dir,
            file_name = normal_data_name_test
        )

    testing_normal_data = torch.tensor(
        AnomalyDetector.normalize(testing_normal_data))

    if args.dummydata:
        testing_abnormal_data = loaddata.load_abnormal_dummydata()
    else:
        testing_abnormal_data = loaddata.load_data_all(
            data_dir = abnormal_data_dir,
            file_name = abnormal_data_name
        )

    testing_abnormal_data = torch.tensor(AnomalyDetector.normalize(testing_abnormal_data))
    print("testing_abnormal_data.shape ", testing_abnormal_data.shape)

    if gpu:
        AnomalyDetector = AnomalyDetector.cuda()
        testing_normal_data = testing_normal_data.cuda()
        testing_abnormal_data = testing_abnormal_data.cuda()

    true_label_normal = np.zeros(len(testing_normal_data) - AnomalyDetector.RED_collection_len * AnomalyDetector.RED_points - 1)
    true_label_abnormal = np.ones(len(testing_abnormal_data) - AnomalyDetector.RED_collection_len * AnomalyDetector.RED_points - 1)
    true_label = np.concatenate((true_label_normal, true_label_abnormal), axis=0)

    pred_normal, p_values_normal = AnomalyDetector.predict(
        testing_normal_data,
        gpu,
        debug=args.debug
        )

    if args.debug:
        feature_idx = 0

        # debug_pred_normal is of size [seq_len-1, batch(=1), features]
        RE_normal, debug_pred_normal = AnomalyDetector._get_reconstruction_error(
            testing_normal_data,
            gpu=gpu)

        seq_dict = {
            "truth": testing_normal_data[1:,feature_idx].detach().numpy(),
            "pred": debug_pred_normal[:,0, feature_idx].detach().numpy(),
        }
        seq_dict["diff"] = (seq_dict["pred"] - seq_dict["truth"])**2
        utils.plot_seq(seq_dict, title="Testing normal prediction")

        # debug_pred_normal is of size [seq_len-1, batch(=1), features]
        RE_abnormal, debug_pred_abnormal = AnomalyDetector._get_reconstruction_error(
            testing_abnormal_data,
            gpu=gpu
            )

        seq_dict = {
            "truth": testing_abnormal_data[1:,feature_idx].detach().numpy(),
            "pred": debug_pred_abnormal[:,0, feature_idx].detach().numpy(),
        }
        seq_dict["diff"] = (seq_dict["pred"] - seq_dict["truth"])**2
        utils.plot_seq(seq_dict, title="Testing abnormal prediction")

        # debug_ref is of size [seq_len-1, batch(=1), features]
        RE_ref, debug_ref = AnomalyDetector._get_reconstruction_error(
            ref_normal_data,
            gpu=gpu)

        seq_dict = {
            "truth": ref_normal_data[1:,feature_idx].detach().numpy(),
            "pred": debug_ref[:,0, feature_idx].detach().numpy(),
            }
        seq_dict["diff"] = (seq_dict["pred"] - seq_dict["truth"])**2
        utils.plot_seq(seq_dict, title="Train normal ref prediction")

        RE_seq_dict = {
            "RE_reference": RE_ref,
            "RE_normal": RE_normal,
            "RE_abnormal": RE_abnormal
        }
        utils.plot_seq(RE_seq_dict, title="Reconstruction errors")
        utils.plot_cdf(RE_seq_dict, title="RED cdf")


    print("p_values_normal.shape ", len(p_values_normal))
    print("p_values_normal.mean ", np.mean(p_values_normal))

    pred_abnormal, p_values_abnormal = AnomalyDetector.predict(
        testing_abnormal_data,
        gpu,
        debug=args.debug
        )
    print("p_values_abnormal.shape ", len(p_values_abnormal))
    print("p_values_abnormal.mean ", np.mean(p_values_abnormal))

    pred = np.concatenate((pred_normal, pred_abnormal), axis=0)
    pred_score = np.concatenate((p_values_normal, p_values_abnormal), axis=0)
    print("true_label.shape", true_label.shape, "pred.shape", pred.shape)

    tp, fp, fn, tn, acc, prec, rec, f1, fpr, tpr, thresholds, roc_auc = (
        utils.eval_metrics(
            truth = true_label,
            pred = pred,
            anomaly_score = -np.log10(pred_score+1e-50)  # Anomaly score=-log(p_value)
            )
    )

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of LSTM anomaly detector')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc

if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        args = utils.create_parser()

        np.random.seed(0)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if args.debug:
            print(args)

        eval_detector(args)

    except SystemExit:
        sys.exit(0)

    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
