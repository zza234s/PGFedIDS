import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from collections import defaultdict
import math
from sklearn.preprocessing import label_binarize
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import metrics
from pycm import *
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader, Subset
from torch.functional import F
class clientLTIIDS_Plus(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()

        self.lamda = args.lamda

        self.cuda_cka = CudaCKA(args.device)
        self.lam = torch.tensor([1.0], requires_grad=True, device=self.device)
        self.gamma = torch.tensor([1.0], requires_grad=True, device=self.device)
        self.optimizer_learned_weight_for_inference = torch.optim.Adam([self.lam, self.gamma], lr=args.ensemble_lr)
        # self.optimizer_learned_weight_for_inference = torch.optim.SGD([self.lam], lr=1e-2)

        self.use_ensemble =args.use_ensemble
        self.test_acc = []
    def load_adaptive_data(self, batch_size=None, ratio=0.1):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        subset_size = int(len(train_data) * 0.1)
        indices = torch.randperm(len(train_data))[:subset_size]
        subset = Subset(train_data, indices)
        adaptive_dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        return adaptive_dataloader

    def train_for_ensemble_weight(self):
        adaptive_loader =self.load_adaptive_data()
        self.model.eval()

        global_ptorotypes = torch.stack(list(self.global_protos.values()))
        for i, (x, y) in enumerate(adaptive_loader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            rep = self.model.base(x)
            local_model_output = self.model.head(rep)

            # calculate the similarity between the local prototypes and the global prototypes
            rep_norm = rep / rep.norm(dim=1, keepdim=True)
            prototype_norm = global_ptorotypes / global_ptorotypes.norm(dim=1, keepdim=True)
            sim = torch.matmul(rep_norm, prototype_norm.T)


            normal_local_output = F.softmax(local_model_output, dim=1)
            normal_prototype_output = F.softmax(sim, dim=1)

            # ensemble output
            output = self.lam * normal_local_output + (1-self.lam) * normal_prototype_output

            loss = self.loss(output, y)
            self.optimizer_learned_weight_for_inference.zero_grad()
            loss.backward()
            self.optimizer_learned_weight_for_inference.step()

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()

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

        self.collect_protos()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        reps_dict = defaultdict(list)
        agg_local_protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                owned_classes = y.unique()
                for cls in owned_classes:
                    filted_reps = rep[y == cls].detach()
                    reps_dict[cls.item()].append(filted_reps.data)

                # for i, yy in enumerate(y):
                #     y_c = yy.item()
                #     protos[y_c].append(rep[i, :].detach().data)
            for cls, protos in reps_dict.items():
                mean_proto = torch.cat(protos).mean(dim=0)
                agg_local_protos[cls] = mean_proto

        # 对于未见类，生成全zero tensor上传server
        unseen_classes = set(range(self.num_classes)) - set(agg_local_protos.keys())
        if unseen_classes != set():
            print(
                f'client{self.id}: class(es) {unseen_classes} not in the local training dataset, add zero tensors to local prototype')
        for label in range(self.num_classes):
            if label not in agg_local_protos:
                agg_local_protos[label] = torch.zeros(list(agg_local_protos.values())[0].shape[0],
                                                      device=self.device)

        agg_local_protos = dict(sorted(agg_local_protos.items()))  # 确保上传的dict按key升序排序
        self.protos = agg_local_protos

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()
        lam = self.lam.detach().data
        gamma = self.gamma.detach().data

        correct_num = 0
        test_num = 0

        correct_prototype =0
        correct_local = 0

        y_prob = []
        y_true = []
        y_pred = []
        y_true_list = []

        if self.global_protos is not None:
            global_ptorotypes = torch.stack(list(self.global_protos.values()))  # dict to tensor

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                local_model_output = self.model.head(rep)

                if self.global_protos is not None and self.use_ensemble:
                    # prototype_sim = self.cuda_cka.linear_CKA(rep.detach().data, global_ptorotypes) #todo 找到支持广播的Linea CKA
                    rep_norm = rep/rep.norm(dim=1, keepdim=True)
                    prototype_norm = global_ptorotypes/global_ptorotypes.norm(dim=1, keepdim=True)
                    sim = torch.matmul(rep_norm, prototype_norm.T)
                    # Normalize the similarity matrix according to the L1 norm
                    # prototype_based_output = sim / (sim.norm(p=1, dim=1, keepdim=True)+1e-5)
                    # ensemble output
                    normal_local_output = F.softmax(local_model_output, dim=1)
                    normal_prototype_output = F.softmax(sim, dim=1)
                    correct_prototype += (torch.sum(torch.argmax(normal_prototype_output, dim=1) == y)).item()
                    output = self.lam * normal_local_output + self.gamma * normal_prototype_output
                    # output = self.lam * local_model_output + (1-self.lam)* prototype_based_output  # ensemble_out
                    # output = lam * local_model_output + gamma * prototype_based_output #ensemble_out
                else:
                    output = local_model_output
                correct_local += (torch.sum(torch.argmax(local_model_output, dim=1) == y)).item()
                correct_num += (torch.sum(torch.argmax(output, dim=1) == y)).item()
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

        test_acc = correct_num / test_num * 1.0
        self.test_acc.append(test_acc)
        return correct_num, test_num, auc, indicators, correct_local, correct_prototype, lam, gamma

class CudaCKA(object):
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)
