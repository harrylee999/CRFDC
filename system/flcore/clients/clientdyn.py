import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms


class clientDyn(Client):
    def __init__(self, args, id, train_samples,  **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

        self.alpha = args.alpha

        self.global_model_vector = None
        old_grad = copy.deepcopy(self.model)
        old_grad = model_parameter_vector(old_grad)
        self.old_grad = torch.zeros_like(old_grad)
        

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

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

                if self.global_model_vector != None:
                    v1 = model_parameter_vector(self.model)
                    loss += self.alpha/2 * torch.norm(v1 - self.global_model_vector, 2)
                    loss -= torch.dot(v1, self.old_grad)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


        if self.global_model_vector != None:
            v1 = model_parameter_vector(self.model).detach()
            self.old_grad = self.old_grad - self.alpha * (v1 - self.global_model_vector)

        print(f"client{self.id} train_samples:{self.train_samples} train cost time :{time.time() - start_time}")

        # self.model.cpu()




    def set_parameters(self, model):
        # for new_param, old_param in zip(model.parameters(), self.model.parameters()):
        #     old_param.data = new_param.data.clone()
        self.global_model_vector = model_parameter_vector(model).detach().clone()
        self.model.load_state_dict(copy.deepcopy(model.state_dict()))
       

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)

                if self.global_model_vector != None:
                    v1 = model_parameter_vector(self.model)
                    loss += self.alpha/2 * torch.norm(v1 - self.global_model_vector, 2)
                    loss -= torch.dot(v1, self.old_grad)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num


def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)