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
import torch
import sys
import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
import os
import h5py
class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.re_IDS_indices=[]


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            train_start_time = time.time()
            for client in self.selected_clients:
                client.train()

            train_end_time = time.time()
            train_times = train_end_time - train_start_time
            averaging_train_time =train_times/len(self.selected_clients)
            print("client averaging train time: ",averaging_train_time)


            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        print('###########################################')
        total_inference_cost = 0
        total_train_cost = 0
        for c in self.selected_clients:
            total_inference_cost+= sum(c.inference_cost)/len(c.inference_cost)
            total_train_cost += c.train_time_cost['total_cost'] / c.train_time_cost['num_rounds']
        print("\nAverage training time cost per client per round.")
        print(total_train_cost/len(self.selected_clients))
        print("\nAverage inference time per client per round:")
        print(total_inference_cost/len(self.selected_clients))

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]

        indices = np.mean(np.stack(stats[4]),axis=0)
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))

        ###########################
        self.re_IDS_indices.append(indices)
        self.res_std.append(np.std(accs))
    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_indicators = []
        for c in self.clients:
            ct, ns, auc, indicators = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            tot_indicators.append(indicators)
        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, tot_indicators

    # def save_results(self):
    #     algo = self.dataset + "_" + self.algorithm
    #     result_path = "../results/"
    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)
    #
    #     if (len(self.rs_test_acc)):
    #         algo = algo + "_" + self.goal + "_" + str(self.times)
    #         file_path = result_path + "{}.h5".format(algo)
    #         print("File path: " + file_path)
    #
    #         with h5py.File(file_path, 'w') as hf:
    #             hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
    #             hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
    #             hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
    #             hf.create_dataset('re_IDS_indices', data=self.re_IDS_indices)


# #zhl: 为了更准确地计算传参量
# def get_size(obj, seen=None):
#     # 用于存储已经检查过的对象，避免重复计算
#     if seen is None:
#         seen = set()
#
#     # 获取对象的id
#     obj_id = id(obj)
#
#     # 如果对象已经检查过，直接返回0
#     if obj_id in seen:
#         return 0
#
#     # 将对象id添加到已检查集合中
#     seen.add(obj_id)
#
#     # size = sys.getsizeof(obj)
#     size=0
#     # 如果对象是字典，递归计算其键和值的大小
#     # 如果对象是PyTorch张量，获取其内存开销
#     if isinstance(obj, torch.Tensor):
#         size += obj.nelement()
#         # size += obj.element_size() * obj.nelement()
#
#     elif isinstance(obj, dict):
#         size += sum([get_size(k, seen) + get_size(v, seen) for k, v in obj.items()])
#
#     # 如果对象是列表或元组，递归计算其元素的大小
#     elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
#         size += sum([get_size(item, seen) for item in obj])
#
#
#     return size