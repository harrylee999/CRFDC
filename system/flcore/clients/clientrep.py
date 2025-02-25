import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
import copy

class clientRep(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples,  **kwargs)
        
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)

        self.plocal_steps = args.local_epochs
        self.pmodel = copy.deepcopy(self.model)

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True


        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        
        for step in range(self.plocal_steps):
            for i, (x, y) in enumerate(trainloader):
            
                x = x.to(self.device)
                y = y.to(self.device)
                x = transform_train(x)


                self.optimizer_per.zero_grad()
                _,output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer_per.step()
        self.pmodel = copy.deepcopy(self.model)
                
        max_local_steps = self.local_epochs

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

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


        print(f"client{self.id} train_samples:{self.train_samples} train cost time :{time.time() - start_time}")
        


    def fine_tune(self):
        trainloader = self.load_train_data()
        
        # self.model.to(self.device)
        self.model.train()

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True


        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        
        for step in range(self.plocal_steps):
            for i, (x, y) in enumerate(trainloader):
            
                x = x.to(self.device)
                y = y.to(self.device)
                x = transform_train(x)


                self.optimizer_per.zero_grad()
                _,output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer_per.step()



            
    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()