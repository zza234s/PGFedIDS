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

import time
import torch
import torch.nn as nn
from flcore.clients.clientgh import clientGH
from flcore.servers.serverbase import Server
from threading import Thread
from torch.utils.data import DataLoader
from thop import profile
from thop import clever_format
import sys
class FedGH(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.global_model = None

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGH)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate

        self.head = self.clients[0].model.head
        # self.opt_h = torch.optim.SGD(self.head.parameters(), lr=self.server_learning_rate)
        self.opt_h = torch.optim.Adam(self.head.parameters(), lr=self.server_learning_rate)
        self.total_prototye_bytes=0.0

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()
                client.collect_protos()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()
            self.train_head()
            ##

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
        print("\n Average prototype size: ")
        print(self.total_prototye_bytes/100/len(self.selected_clients))

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.head)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.total_prototye_bytes+= get_size(torch.stack(list(client.protos.values())))
            for cc in client.protos.keys():
                y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                self.uploaded_protos.append((client.protos[cc], y))
            
    def train_head(self):
        proto_loader = DataLoader(self.uploaded_protos, self.batch_size, drop_last=False, shuffle=True)

        for p, y in proto_loader:
            out = self.head(p)
            loss = self.CEloss(out, y)
            self.opt_h.zero_grad()
            loss.backward()
            self.opt_h.step()

def get_size(obj, seen=None):
    # 用于存储已经检查过的对象，避免重复计算
    if seen is None:
        seen = set()

    # 获取对象的id
    obj_id = id(obj)

    # 如果对象已经检查过，直接返回0
    if obj_id in seen:
        return 0

    # 将对象id添加到已检查集合中
    seen.add(obj_id)

    size = 0.0

    # 如果对象是字典，递归计算其键和值的大小
    # 如果对象是PyTorch张量，获取其内存开销
    if isinstance(obj, torch.Tensor):
        size += obj.nelement()

    elif isinstance(obj, dict):
        size += sum([get_size(k, seen) + get_size(v, seen) for k, v in obj.items()])

    # 如果对象是列表或元组，递归计算其元素的大小
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(item, seen) for item in obj])


    return size