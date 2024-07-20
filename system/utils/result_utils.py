# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import h5py
import numpy as np
import os
from collections import defaultdict
import pandas as pd

# def average_data(algorithm="", dataset="", goal="", times=10):
#     test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)
#
#     max_accurancy = []
#     for i in range(times):
#         max_accurancy.append(test_acc[i].max())
#
#     print("std for best accurancy:", np.std(max_accurancy))
#     print("mean for best accurancy:", np.mean(max_accurancy))

def average_data(algorithm="", dataset="", goal="", times=10, metrics=['rs_test_acc'], best_res_key='rs_test_acc'):
    result = get_all_results_for_one_algo_specified_metrics(algorithm, dataset, goal, times, metrics)

    max_metrics = defaultdict(list)
    bset_rounds = []

    assert best_res_key in metrics
    assert len(result[best_res_key][0].shape) == 1
    for i in range(times):
        max_metrics[best_res_key].append(result[best_res_key][i].max())
        bset_rounds.append(result[best_res_key][i].argmax())
    print(f"std for best {best_res_key} (best res key):", np.std(max_metrics[best_res_key]))
    print(f"mean for best {best_res_key} (best res key):", np.mean(max_metrics[best_res_key]))

    # 保存 bese_key对应指标至txt文件
    txt_file_path = f"../results/{dataset}_{algorithm}_{goal}_{best_res_key}.txt"
    os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
    with open(txt_file_path, 'w') as f:
        f.write(f"std for best {best_res_key}: {np.std(max_metrics[best_res_key])}\n")
        f.write(f"mean for best {best_res_key}: {np.mean(max_metrics[best_res_key])}\n")


    for metric in metrics:
        if metric == best_res_key:
            continue
        print("=" * 50)
        shape_dim = len(result[metric][0].shape)
        for i in range(times):
            if shape_dim == 1:
                max_metrics[metric].append(result[metric][i].max())
            else:
                max_metrics[metric].append(result[metric][i][bset_rounds[i]])

        if shape_dim == 1:
            std_value = np.std(max_metrics[metric])
            mean_value = np.mean(max_metrics[metric])
            print(f"std for best {metric}:", std_value)
            print(f"mean for best {metric}:", mean_value)
            txt_file_path = f"../results/{dataset}_{algorithm}_{goal}_{metric}.txt"
            os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
            with open(txt_file_path, 'w') as f:
                f.write(f"std for best {metric}: {std_value}\n")
                f.write(f"mean for best {metric}: {mean_value}\n")
        else:
            print(f"std for best {metric}:\n", np.std(max_metrics[metric], axis=0))
            print(f"mean for best {metric}:\n", np.mean(max_metrics[metric], axis=0))
            csv_file_name= "../results/"+ dataset + "_" + algorithm + "_" + goal + "_" + metric +'.csv'
            mean_data =np.mean(max_metrics[metric],axis=0)
            std_data =  np.std(max_metrics[metric], axis=0)
            combined_array = np.column_stack((mean_data, std_data))

            df=pd.DataFrame(combined_array.T,
                            columns=['macro_precision', 'weighted_precision', 'macro_recall', 'weighted_recall', 'macro_f1','weighted_f1']
                            )
            print(mean_data)
            df.to_csv(csv_file_name)

def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def get_all_results_for_one_algo_specified_metrics(algorithm="", dataset="", goal="", times=10,
                                                   metrics=['rs_test_acc']):
    result = defaultdict(list)
    algorithms_list = [algorithm] * times
    for metric in metrics:
        for i in range(times):
            file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
            result[metric].append(np.array(read_data_then_delete(file_name, delete=False, metric=metric)))

    return result


def read_data_then_delete(file_name, delete=False, metric=None):
    file_path = "../results/" + file_name + ".h5"

    if not metric:
        metric = 'rs_test_acc'

    with h5py.File(file_path, 'r') as hf:
        result = np.array(hf.get(metric))

    if delete:
        os.remove(file_path)

    print(f"{metric} Length: ", len(result))

    return result


def plot_acc_curve(clients):
    import math
    import matplotlib.pyplot as plt
    # matplotlib.use('TkAgg')

    client_num = len(clients)
    if client_num > 5:
        nrows = 2
        ncols = math.ceil(client_num // nrows)
    else:
        nrows = 1
        ncols = client_num
    # fig, axs = plt.subplots(1, client_num, figsize=(12, 4), sharex=True, sharey=True)
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 4), sharex=True, sharey=True)
    for client in clients:
        acc = client.test_acc
        c_id = client.id
        idx = c_id
        x = np.arange(1, len(acc) + 1)
        y = np.array(acc)
        if nrows == 1:
            axs[idx].plot(x, y, label=f'test_acc')
            axs[idx].xlabel = 'epoch'
            axs[idx].ylabel = 'accuracy'
            axs[idx].set_title(f'Client {c_id}')
            axs[idx].legend()
        else:
            axs[idx // ncols][idx % ncols].plot(x, y, label=f'test_acc')
            axs[idx // ncols][idx % ncols].xlabel = 'epoch'
            axs[idx // ncols][idx % ncols].ylabel = 'accuracy'
            axs[idx // ncols][idx % ncols].set_title(f'Client {c_id}')
            axs[idx // ncols][idx % ncols].legend()
    plt.show()