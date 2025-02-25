import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data
from utils.ALA import ALA
from torchvision.transforms import transforms


class clientALA(Client):
    def __init__(self, args, id, train_samples,  **kwargs):
        super().__init__(args, id, train_samples,  **kwargs)

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx

        train_data = read_client_data(self.dataset, self.id)
        self.ALA = ALA(self.id, self.loss, train_data, self.batch_size, 
                    self.rand_percent, self.layer_idx, self.eta, self.device)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs


        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
       
                x = x.to(self.device)
                y = y.to(self.device)
                x = transform_train(x)
               
                _,output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        print(f"client{self.id} train_samples:{self.train_samples} train cost time :{time.time() - start_time}")



    def local_initialization(self, received_global_model):
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)