# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification
# https://arxiv.org/abs/2102.00457
import argparse
import os
import platform
import socket
import time
from datetime import datetime
import mat73
import scipy.io as sio

import numba
import numpy as np
import pandas as pd
import psutil
import pytz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sklearn.utils import shuffle

from multirocket.multirocket_multivariate import MultiRocket
from utils.data_loader import process_ts_data
from utils.tools import create_directory

pd.set_option('display.max_columns', 500)

itr = 0
num_features = 10000
save = True
num_threads = 0

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datapath", type=str, required=False, default='D:/EEG_data_downsample/8/')#'D:/EEG_data_overlap/1024_64/') #"D:/EEG_data_Visual/")
parser.add_argument("-p", "--problem", type=str, required=False, default="ArticularyWordRecognition")
parser.add_argument("-i", "--iter", type=int, required=False, default=0)
parser.add_argument("-n", "--num_features", type=int, required=False, default=50000)
parser.add_argument("-t", "--num_threads", type=int, required=False, default=8)
parser.add_argument("-s", "--save", type=bool, required=False, default=True)
parser.add_argument("-v", "--verbose", type=int, required=False, default=2)

arguments = parser.parse_args()

if __name__ == '__main__':
    data_path = arguments.datapath
    problem = arguments.problem
    num_features = arguments.num_features
    num_threads = arguments.num_threads
    itr = arguments.iter
    save = arguments.save
    verbose = arguments.verbose
    seed = 100

    window_length = [128, 256, 512, 1024]
    stride_length = [32, 64, 128, 256, 512]
    sub_str =['S01', 'S02', 'S09', 'S18', 'S24', 'S28', 'S29']
    task_list = ['VS','VI', 'AS', 'AI']

    output_path = os.getcwd() + "/output/Downsample/8/"
    classifier_name = "MultiRocket_{}".format(num_features)

    for task in task_list:

        data_path = arguments.datapath + task + '/'
        print(data_path)
        for sub in sub_str:
            problem = sub

            data_folder = data_path + problem + '/'
            print(data_folder)

            if os.path.exists(data_folder):
                if num_threads > 0:
                    numba.set_num_threads(num_threads)
                print(os.getcwd())

                start = time.perf_counter()

                output_dir = "{}/multirocket/resample_{}/{}/{}/".format(
                    output_path,
                    itr,
                    classifier_name,
                    task
                )
                if save:
                    create_directory(output_dir)

                print("=======================================================================")
                print("Starting Experiments")
                print("=======================================================================")
                print("Data path: {}".format(data_path))
                print("Output Dir: {}".format(output_dir))
                print("Iteration: {}".format(itr))
                print("Problem: {}".format(problem))
                print("Number of Features: {}".format(num_features))

                file_name = 'data.mat'

                print("Loading data")
                X_test = mat73.loadmat(data_folder + file_name)


                #for original
                y_train = X_test['label_training']
                y_test = X_test['label_test_total']
                X_train = X_test['training_data']
                X_test = X_test['test_data_total']

                y_train = np.array(y_train)
                y_test = np.array(y_test)
                X_train = np.array(X_train)
                X_test = np.array(X_test)

                X_train = X_train[:,0:32,:]
                X_test = X_test[:,0:32,:]

                X_train, y_train = shuffle(X_train, y_train, random_state=seed)
                X_test, y_test = shuffle(X_test, y_test, random_state=seed)

                # X_train, y_train = load_from_tsfile_to_dataframe(train_file)
                # X_test, y_test = load_from_tsfile_to_dataframe(test_file)
                #
                # X_train = process_ts_data(X_train, normalise=False)
                # X_test = process_ts_data(X_test, normalise=False)

                print(type(X_train))
                print(X_train.shape)
                print(np.unique(y_train))

                nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

                classifier = MultiRocket(
                    num_features=num_features,
                    verbose=verbose
                )
                yhat_train = classifier.fit(
                    X_train, y_train,
                    predict_on_train=True
                )

                if yhat_train is not None:
                    train_acc = accuracy_score(y_train, yhat_train)
                else:
                    train_acc = -1

                yhat_test = classifier.predict(X_test)
                test_acc = accuracy_score(y_test, yhat_test)

                sio.savemat(output_dir + sub+ 'classfied.mat', {'predicted_label':yhat_test, 'true_label':y_test})

                # get cpu information
                physical_cores = psutil.cpu_count(logical=False)
                logical_cores = psutil.cpu_count(logical=True)
                cpu_freq = psutil.cpu_freq()
                max_freq = cpu_freq.max
                min_freq = cpu_freq.min
                memory = np.round(psutil.virtual_memory().total / 1e9)

                df_metrics = pd.DataFrame(data=np.zeros((1, 21), dtype=np.float), index=[0],
                                          columns=['timestamp', 'itr', 'classifier',
                                                   'num_features',
                                                   'dataset',
                                                   'train_acc', 'train_time',
                                                   'test_acc', 'test_time',
                                                   'generate_kernel_time',
                                                   'apply_kernel_on_train_time',
                                                   'apply_kernel_on_test_time',
                                                   'train_transform_time',
                                                   'test_transform_time',
                                                   'machine', 'processor',
                                                   'physical_cores',
                                                   "logical_cores",
                                                   'max_freq', 'min_freq', 'memory'])
                df_metrics["timestamp"] = datetime.utcnow().replace(tzinfo=pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
                df_metrics["itr"] = itr
                df_metrics["classifier"] = classifier_name
                df_metrics["num_features"] = num_features
                df_metrics["dataset"] = problem
                df_metrics["train_acc"] = train_acc
                df_metrics["train_time"] = classifier.train_duration
                df_metrics["test_acc"] = test_acc
                df_metrics["test_time"] = classifier.test_duration
                df_metrics["generate_kernel_time"] = classifier.generate_kernel_duration
                df_metrics["apply_kernel_on_train_time"] = classifier.apply_kernel_on_train_duration
                df_metrics["apply_kernel_on_test_time"] = classifier.apply_kernel_on_test_duration
                df_metrics["train_transform_time"] = classifier.train_transforms_duration
                df_metrics["test_transform_time"] = classifier.test_transforms_duration
                df_metrics["machine"] = socket.gethostname()
                df_metrics["processor"] = platform.processor()
                df_metrics["physical_cores"] = physical_cores
                df_metrics["logical_cores"] = logical_cores
                df_metrics["max_freq"] = max_freq
                df_metrics["min_freq"] = min_freq
                df_metrics["memory"] = memory

                print(df_metrics)
                if save:
                    df_metrics.to_csv(output_dir + sub+ '_results.csv', index=False)
            else:
                print('path does not exist')