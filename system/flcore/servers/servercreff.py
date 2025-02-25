import time
from flcore.clients.clientcreff import clientCreff
from flcore.servers.serverbase import Server
from tqdm import tqdm
import torch
import numpy as np
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import copy
from utils.data_utils import TensorDataset
from torch.utils.data.dataloader import DataLoader

class FedCreff(Server):
    def __init__(self, args):
        super().__init__(args)
        self.num_fea = args.num_fea
        self.fea_dim = args.fea_dim
        self.fea_lr = args.fea_lr
        self.crt_epoch = args.crt_epoch
        self.match_epoch = args.match_epoch
        self.cr_batch_size = args.cr_batch_size

        # select slow clients
        self.set_clients(args, clientCreff)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.feature_syn = torch.randn(size=(self.num_classes * self.num_fea, self.fea_dim), dtype=torch.float,requires_grad=True, device=args.device)
        self.label_syn = torch.tensor([np.ones(self.num_fea) * i for i in range(self.num_classes)], dtype=torch.long,
                                      requires_grad=False, device=self.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        self.optimizer_feature = SGD([self.feature_syn, ], lr=args.lr_feature,weight_decay=1e-5,momentum=0.9)  # optimizer_img for synthetic data
        self.criterion = CrossEntropyLoss().to(args.device)
        self.feature_net = nn.Linear(self.fea_dim, self.num_classes).to(args.device)



    
    def train(self):
        temp_model = nn.Linear(self.fea_dim, self.num_classes).to(self.device)
        syn_params = temp_model.state_dict()

        for i in tqdm(range(1, self.global_rounds+1), desc='server-training'):
            s_t = time.time()

            global_params = self.global_model.state_dict()
            syn_feature_params = copy.deepcopy(global_params)
            syn_feature_params['head.bias'] = syn_params['bias']
            syn_feature_params['head.weight'] = syn_params['weight']

            self.selected_clients = self.select_clients()
            self.send_models()
            list_clients_gradient = []

            print()
            for client in self.selected_clients:
                truth_gradient = client.compute_gradient(copy.deepcopy(syn_feature_params))
                list_clients_gradient.append(copy.deepcopy(truth_gradient))
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.update_feature_syn(copy.deepcopy(syn_feature_params), list_clients_gradient)
            syn_params, ft_params = self.feature_re_train(copy.deepcopy(self.global_model.state_dict()), self.cr_batch_size )


            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")
            self.evaluate()
            print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)

            if i%self.save_gap == 0:
                self.save_results(i)

            if self.current_acc < self.test_acc[-1]:
                self.current_acc = self.test_acc[-1]
                torch.save(self.global_model.state_dict(),"../results/"+ self.dataset+"/model/"+self.algorithm+"best.pt")
    def get_acc(self):
        result_path = "../results/"+ self.dataset+"/model/"
        self.global_model.load_state_dict(torch.load(result_path+self.algorithm+"best.pt"))
        self.evaluate()


    def update_feature_syn(self, global_params, list_clients_gradient):
        feature_net_params = self.feature_net.state_dict()

        feature_net_params['bias'] = global_params['head.bias']
        feature_net_params['weight'] = global_params['head.weight']

        self.feature_net.load_state_dict(feature_net_params)
        self.feature_net.train()
        net_global_parameters = list(self.feature_net.parameters())
        gw_real_all = {class_index: [] for class_index in range(self.num_classes)}
        for gradient_one in list_clients_gradient:
            for class_num, gradient in gradient_one.items():
                gw_real_all[class_num].append(gradient)
        gw_real_avg = {class_index: [] for class_index in range(self.num_classes)}
        # aggregate the real feature gradients
        for i in range(self.num_classes):
            gw_real_temp = []
            list_one_class_client_gradient = gw_real_all[i]

            if len(list_one_class_client_gradient) != 0:
                weight_temp = 1.0 / len(list_one_class_client_gradient)
                for name_param in range(2):
                    list_values_param = []
                    for one_gradient in list_one_class_client_gradient:
                        list_values_param.append(one_gradient[name_param] * weight_temp)
                    value_global_param = sum(list_values_param)
                    gw_real_temp.append(value_global_param)
                gw_real_avg[i] = gw_real_temp
        # update the federated features.
        for ep in range(self.match_epoch):
            loss_feature = torch.tensor(0.0).to(self.device)
            for c in range(self.num_classes):
                if len(gw_real_avg[c]) != 0:
                    feature_syn = self.feature_syn[c * self.num_fea:(c + 1) * self.num_fea].reshape((self.num_fea, self.fea_dim))
                    lab_syn = torch.ones((self.num_fea,), device=self.device, dtype=torch.long) * c
                    output_syn = self.feature_net(feature_syn)
                    loss_syn = self.criterion(output_syn, lab_syn)
                    # compute the federated feature gradients of class c
                    gw_syn = torch.autograd.grad(loss_syn, net_global_parameters, create_graph=True)
                    loss_feature += match_loss(gw_syn, gw_real_avg[c],self.device)
            self.optimizer_feature.zero_grad()
            loss_feature.backward()
            self.optimizer_feature.step()

    def feature_re_train(self, fedavg_params, batch_size_local_training):
        feature_syn_train_ft = copy.deepcopy(self.feature_syn.detach())
        label_syn_train_ft = copy.deepcopy(self.label_syn.detach())
        dst_train_syn_ft = TensorDataset(feature_syn_train_ft, label_syn_train_ft)
        ft_model = nn.Linear(self.fea_dim, self.num_classes).to(self.device)
        optimizer_ft_net = SGD(ft_model.parameters(), lr=self.fea_lr,weight_decay=1e-5,momentum=0.9)  # optimizer_img for synthetic data
        ft_model.train()
        for epoch in range(self.crt_epoch):
            trainloader_ft = DataLoader(dataset=dst_train_syn_ft,
                                        batch_size=batch_size_local_training,
                                        shuffle=True)
            for data_batch in trainloader_ft:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = ft_model(images)
                loss_net = self.criterion(outputs, labels)
                optimizer_ft_net.zero_grad()
                loss_net.backward()
                optimizer_ft_net.step()
        ft_model.eval()
        feature_net_params = ft_model.state_dict()

        fedavg_params['head.bias'] = feature_net_params['bias']

        fedavg_params['head.weight'] = feature_net_params['weight']
  
        return copy.deepcopy(ft_model.state_dict()), copy.deepcopy(fedavg_params)

def match_loss(gw_syn, gw_real, device):
    dis = torch.tensor(0.0).to(device)

    for ig in range(len(gw_real)):
        gwr = gw_real[ig]
        gws = gw_syn[ig]
        dis += distance_wb(gwr, gws)
    return dis


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        # return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis