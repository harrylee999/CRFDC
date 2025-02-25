import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.param_aug import ParamDiffAug,DiffAugment
from utils.data_utils import read_client_json,read_client_data,remix,read_total_json
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

class clientCreff(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        self.dsa_param = ParamDiffAug()
        self.data_client = read_client_data(self.dataset, self.id)
        self.total_class_num = read_total_json(self.dataset)
        

    def train(self):
        trainloader = DataLoader(self.data_client, self.batch_size, shuffle=True,pin_memory=True)

        # self.model.to(self.device)
        self.model.train()

        
        start_time = time.time()

        max_local_steps = self.local_epochs
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                x = transform_train(x)
                #mixed_image,label_a,label_b,l_list = remix(x,y,self.total_class_num,self.device)
                self.optimizer.zero_grad()
                _,output = self.model(x)
                # loss = l_list * self.loss(output, label_a) + (1 - l_list) * self.loss(output, label_b)
                # loss = loss.mean()
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()
        print(f"client{self.id} train_samples:{self.train_samples} train cost time :{time.time() - start_time}")

    def compute_gradient(self, global_params):
        list_class = []
        per_class_compose = []
        client_dict_per_class = read_client_json(self.dataset,self.id)[1]
        for c in client_dict_per_class:
            if c[1]!=0:
                list_class.append(c[0])
                per_class_compose.append(c[1])


        images_all = []
        labels_all = []
        indices_class = {class_index: [] for class_index in list_class}

        images_all = [torch.unsqueeze(self.data_client[i][0], dim=0) for i in range(len(self.data_client))]
        labels_all = [self.data_client[i][1] for i in range(len(self.data_client))]


        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(self.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=self.device)

        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        self.model.load_state_dict(global_params)

        self.model.eval()
        self.model.head.train()
        net_parameters = list(self.model.head.parameters())
        criterion = CrossEntropyLoss().to(self.device)
        # gradients of all classes
        truth_gradient_all = {index: [] for index in list_class}
        truth_gradient_avg = {index: [] for index in list_class}

        # choose to repeat 10 times
        for num_compute in range(10):
            for c, num in zip(list_class, per_class_compose):
                img_real = get_images(c, 32)
                # transform
                seed = int(time.time() * 1000) % 100000
                img_real = DiffAugment(img_real, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=self.dsa_param)
                lab_real = torch.ones((img_real.shape[0],), device=self.device, dtype=torch.long) * c
                feature_real, output_real = self.model(img_real)
                loss_real = criterion(output_real, lab_real)
                # compute the real feature gradients of class c
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                truth_gradient_all[c].append(gw_real)
        for i in list_class:
            gw_real_temp = []
            gradient_all = truth_gradient_all[i]
            weight = 1.0 / len(gradient_all)
            for name_param in range(len(gradient_all[0])):
                list_values_param = []
                for client_one in gradient_all:
                    list_values_param.append(client_one[name_param] * weight)
                value_global_param = sum(list_values_param)
                gw_real_temp.append(value_global_param)
            # the real feature gradients of all classes
            truth_gradient_avg[i] = gw_real_temp
        return truth_gradient_avg


