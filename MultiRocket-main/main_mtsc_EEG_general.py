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

from multirocket.multirocket_multivariate import MultiRocket
from utils.data_loader import process_ts_data
from utils.tools import create_directory

pd.set_option('display.max_columns', 500)

itr = 0
num_features = 10000
save = True
num_threads = 0

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datapath", type=str, required=False, default="D:/EEG_data_Visual/")
parser.add_argument("-p", "--problem", type=str, required=False, default="ArticularyWordRecognition")
parser.add_argument("-i", "--iter", type=int, required=False, default=0)
parser.add_argument("-n", "--num_features", type=int, required=False, default=50000)
parser.add_argument("-t", "--num_threads", type=int, required=False, default=-1)
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
    isloocv = False

    sub_str =['S01', 'S02', 'S09', 'S18', 'S24', 'S28', 'S29']
    sub_str = ['S02']
    task_list = ['VI']#, 'AS', 'AI']
    feature_list = [50000] #[10000,30000,50000,100000,200000,300000,400000,500000]

    output_path = os.getcwd() + "/output_general_ApEn2/"

    for num_features in feature_list:
        classifier_name = "MultiRocket_{}".format(num_features)
        for task in task_list:

            data_path = data_path + task + '/'

            # for original
            if task == 'VS' or task == 'VI':
                data_path = 'D:/EEG_data_Visual/' +task+'/balanced/'
            else:
                data_path = 'D:/EEG_data_Auditory/'+task+'/'

            total_data = []
            total_label = []

            for sub in sub_str:
                problem = sub
                # for original
                if task == 'VS' or task == 'VI':
                    data_folder = data_path + problem + "/"
                else:
                    data_folder = data_path

                # data_folder = data_path + problem + '/'
                print(data_folder)

                if os.path.exists(data_folder):
                    if num_threads > 0:
                        numba.set_num_threads(num_threads)
                    print(os.getcwd())
                    # output_path = os.getcwd() + "/output/"

                    start = time.perf_counter()

                    output_dir = "{}/multirocket/resample_{}/{}/{}/".format(
                        output_path,
                        itr,
                        classifier_name,
                        task
                    )
                    if save:
                        if isloocv:
                            create_directory(output_dir + 'loocv/')
                        else:
                            create_directory(output_dir + 'non-loocv/')

                    print("=======================================================================")
                    print("Starting Experiments")
                    print("=======================================================================")
                    print("Data path: {}".format(data_path))
                    print("Output Dir: {}".format(output_dir))
                    print("Iteration: {}".format(itr))
                    print("Problem: {}".format(problem))
                    print("Number of Features: {}".format(num_features))

                    # set data folder
                    # for orignial
                    if task == 'VS':
                        file_name = 'vs_data_balanced.mat'

                    elif task == 'VI':
                        file_name = 'vi_data_balanced.mat'
                    else:
                        file_name = sub + '.mat'
                    # file_name = 'overlap_data.mat'

                    print("Loading data")
                    temp_data = mat73.loadmat(data_folder + file_name)


                    #for original
                    if task == 'VS':
                        label = temp_data['balanced_vs_cate_label']
                        label_class = temp_data['balanced_vs_label']
                        temp_data = temp_data['balanced_vs_data']
                        # for classwise
                        # label = label_class

                    elif task == 'VI':
                        label = temp_data['balanced_vi_cate_label']
                        label_class = temp_data['balanced_vi_label']
                        temp_data = temp_data['balanced_vi_data']
                        #for classwise
                        label = label_class
                    elif task == 'AS':
                        label = temp_data['label_class']
                        # label_class = temp_data['label_class']
                        temp_data = temp_data['AS_data']
                    else:
                        label = temp_data['label_class']
                        # label_class = temp_data['label_class']
                        temp_data = temp_data['AI_data']

                    label = np.array(label)
                    # print(np.shape(label))
                    temp_data = np.array(temp_data)
                    temp_data = temp_data[:,0:32,:]

                    if isloocv:
                        label = label
                    else:
                        if len(total_data):
                            total_data = np.concatenate((total_data, temp_data), axis=0)
                            total_label = np.concatenate((total_label, label), axis=0)
                        else:
                            total_data = temp_data
                            total_label = label
                        # print(np.shape(total_label))
                        # print(np.shape(total_data))

                else:
                    print('path does not exist')

            print(np.shape(total_data))
            print(np.shape(total_label))


            X_train, X_test, y_train, y_test = train_test_split(total_data, total_label, test_size=0.2,
                                                                random_state=seed,
                                                                stratify=total_label)
            split_point = 135
            X_test = temp_data[0:split_point, :, :]
            X_train = temp_data[split_point:split_point + 135, :, :]
            y_test = label[0:split_point]
            y_train = label[split_point:split_point + 135]

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

            if isloocv:
                sio.savemat(output_dir + 'loocv/'+'total_classwise_classfied.mat',
                            {'predicted_label': yhat_test, 'true_label': y_test})
            else:
                sio.savemat(output_dir + 'non-loocv/' + 'total_classwise_classfied.mat',
                            {'predicted_label': yhat_test, 'true_label': y_test})

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
                if isloocv:
                    df_metrics.to_csv(output_dir +'loocv/'+'total_classwise_results.csv', index=False)
                else:
                    df_metrics.to_csv(output_dir + 'non-loocv/'+'total_classwise_results.csv', index=False)