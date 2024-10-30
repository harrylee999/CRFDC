import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client



class clientCCVR(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
 
        self.model.eval()


        start_time = time.time()

        feature_list = [[] for _ in range(self.num_classes)]
        self.mean = []
        self.cov = []


        for i, (x, y) in enumerate(trainloader):
            x = x.to(self.device)
            y = y.to(self.device)
            features,_ = self.model(x)
         
            for index,fea in enumerate(features):
                feature_list[y[index]].append(fea)


        for f in feature_list:
            #print(len(f),end = ' ')
            if len(f)<=1:
                self.mean.append(int(0))
                self.cov.append(int(0))
                continue

            
            concatenated = torch.stack(f, dim=0)
            self.mean.append(torch.mean(concatenated,dim=0).reshape(1,-1))
            self.cov.append(torch.cov(concatenated.T))
          
        del x,y,feature_list,fea,f,concatenated

        # self.model.cpu()
        print(f"client{self.id} train_samples:{self.train_samples} cost time :{time.time() - start_time}")


