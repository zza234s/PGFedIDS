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
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class Local(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        # self.load_model()
        self.Budget = []
        self.avg_train_time_cost=[]

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            train_start_time = time.time()
            for client in self.selected_clients:
                client.train()

            train_end_time = time.time()
            train_times = train_end_time - train_start_time
            averaging_train_time = train_times / len(self.selected_clients)
            print("client averaging train time: ", averaging_train_time)

            self.avg_train_time_cost.append(averaging_train_time)


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

        print("\nAverage training time cost per client per round.")
        print(sum(self.avg_train_time_cost[1:])/len(self.avg_train_time_cost[1:]))

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

        self.save_results()
        self.save_global_model()
