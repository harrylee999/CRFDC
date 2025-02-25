import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from torchvision.transforms import transforms
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_json

class clientROD(Client):
    def __init__(self, args, id, train_samples,  **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
                
        self.head = copy.deepcopy(self.model.head)
        self.opt_head = torch.optim.SGD(self.head.parameters(), lr=self.learning_rate)



        client_dict_per_class = read_client_json(self.dataset,self.id)[1]
        aaa = []
        for i in client_dict_per_class:
            aaa.append(float(i[1]))
        self.sample_per_class = torch.tensor(aaa)
     
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
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                x = transform_train(x)


                _,rep = self.model.base(x)
                out_g = self.model.head(rep)
                loss_bsm = balanced_softmax_loss(y, out_g, self.sample_per_class)
                self.optimizer.zero_grad()
                loss_bsm.backward()
                self.optimizer.step()

                out_p = self.head(rep.detach())
                loss = self.loss(out_g.detach() + out_p, y)
                self.opt_head.zero_grad()
                loss.backward()
                self.opt_head.step()

        # self.model.cpu()
        print(f"client{self.id} train_samples:{self.train_samples} train cost time :{time.time() - start_time}")





# https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
