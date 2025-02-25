import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
from utils.data_utils import remix,read_total_json

class clientProx(Client):
    def __init__(self, args, id, train_samples,  **kwargs):
        super().__init__(args, id, train_samples,  **kwargs)

        self.mu = args.mu

        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.loss = nn.CrossEntropyLoss()
        self.remix = False
        if self.remix:
            self.total_class_num = read_total_json(self.dataset)



    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_epochs

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        for step in range(max_local_steps):
            for x, y in trainloader:

                x = x.to(self.device)
                y = y.to(self.device)
                x = transform_train(x)
                if self.remix:
                    mixed_image,label_a,label_b,l_list = remix(x,y,self.total_class_num,self.device)
                    self.optimizer.zero_grad()
                    _,output = self.model(mixed_image)
                    loss = l_list * self.loss(output, label_a) + (1 - l_list) * self.loss(output, label_b)
                    loss = loss.mean()
                else:
                    self.optimizer.zero_grad()
                    _,output = self.model(x)
                    loss = self.loss(output, y)


                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.model.parameters()):
                    fed_prox_reg += ((self.mu / 2) * torch.norm((param - self.global_params[param_index]))**2)
    
                
                loss += fed_prox_reg

                loss.backward()
                self.optimizer.step()

        # self.model.cpu()
        print(f"client{self.id} train_samples:{self.train_samples} train cost time :{time.time() - start_time}")



    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()


