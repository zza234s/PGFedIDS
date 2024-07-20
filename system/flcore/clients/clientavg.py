import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from pycm import *

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.inference_cost =[]
    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        y_pred = []
        y_true_list = []


        start_time = time.time()
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

                y_pred.extend(np.argmax(output.detach().cpu().numpy(), axis=1).tolist())
                y_true_list.extend(y.tolist())
        end_time = time.time()
        inference_time = end_time-start_time
        self.inference_cost.append(inference_time)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        cm = ConfusionMatrix(y_true_list, y_pred)

        macro_precision = cm.average('PPV', none_omit=True)  # Macro-Precision
        weighted_precision = cm.weighted_average('PPV', none_omit=True)  # Weighted-Precision
        macro_recall = cm.average('TPR', none_omit=True)  # Macro-Recall
        weighted_recall = cm.weighted_average('TPR', none_omit=True)  # Weighted-Recall
        macro_f1 = cm.average('F1', none_omit=True)  # Macro-F1 Score
        weighted_f1 = cm.weighted_average('F1', none_omit=True)  # Weighted-F1 Score

        indicators = np.array(
            [macro_precision, weighted_precision, macro_recall, weighted_recall, macro_f1, weighted_f1])

        # 将None替换为np.nan
        indicators = np.where(indicators == 'None', np.nan, indicators).astype(float)
        # 将np.nan替换为0
        indicators = np.nan_to_num(indicators)

        return test_acc, test_num, auc, indicators