import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms

class clientBABU(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

        self.fine_tuning_steps = args.fine_tuning_steps
        

        for param in self.model.head.parameters():
            param.requires_grad = False



    def train(self):
        # for name, param in self.model.state_dict().items():
        #     if name == "base.bn1.weight":
        #         print(param)


        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_epochs

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):

      
                x = x.to(self.device)
                y = y.to(self.device)
                x = transform_train(x)
                self.optimizer.zero_grad()
                _,output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()



    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def fine_tune(self, which_module=['head']):
        trainloader = self.load_train_data()
        
        start_time = time.time()
        
        self.model.train()

        if 'head' in which_module:
            for param in self.model.head.parameters():
                param.requires_grad = True

        if 'base' not in which_module:
            for param in self.model.base.parameters():
                param.requires_grad = False
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        for step in range(self.fine_tuning_steps):
            for i, (x, y) in enumerate(trainloader):
                
                x = x.to(self.device)
                y = y.to(self.device)
                x = transform_train(x)
                self.optimizer.zero_grad()
                _,output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        self.head_model = copy.deepcopy(self.model.head)

