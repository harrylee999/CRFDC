import torch
import numpy as np
import time
import random
from flcore.clients.clientbase import Client
from utils.data_utils import TensorDataset,read_client_json
from torch.utils.data.dataloader import DataLoader
import copy

def distribution_calibration(feature,head_means, head_cov, m,total_class_num):
    feature = feature.cpu().numpy()

    head_mean = np.stack([t.cpu().numpy() for t in head_means], axis=0)
    head_cov = np.stack([t.cpu().numpy() for t in head_cov], axis=0)

    dist = []
    cos = []
    mix_dist = []
    for i in range(len(head_mean)):
        
        d = np.linalg.norm(feature-head_mean[i])
        dist.append(d)
        cos_sim = np.dot(feature, head_mean[i].ravel()) / (np.linalg.norm(feature) * np.linalg.norm(head_mean[i]))
        cos.append(cos_sim)
        mix_dist.append(cos_sim/d)
        
    index = np.argpartition(mix_dist, -m)[-m:]

    weight = []
    total_head_weight = 0

    sim = 0

    for i in index:
        total_head_weight+=(total_class_num[i][1]*mix_dist[i])

    for i in index:
        weight.append(total_class_num[i][1]*mix_dist[i]/total_head_weight)
        sim+=cos[i]*weight[-1]
    
    mean0 = 0
    cov0= 0
    for i,ind in enumerate(index):
        mean0+=weight[i]*head_mean[ind]
        cov0 += weight[i]*head_cov[ind]

    calibrated_mean = (sim/(1+sim))*mean0+(1/(1+sim))*feature
    calibrated_cov = cov0
    return calibrated_mean, calibrated_cov


class clientCRFDC(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        self.num_fea = args.num_fea
        self.fea_dim = args.fea_dim
        self.fea_lr = args.fea_lr
        self.crt_epoch = args.crt_epoch
        self.global_class_num = None
        self.strat_class = 0
        self.m = args.m
        self.my_model = torch.nn.Linear(self.fea_dim, self.num_classes).to(self.device)
        self.cr_batch_size = args.cr_batch_size
        
        self.global_head = copy.deepcopy(self.model.head)
    def cfmv(self):
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

        self.feature_list = feature_list
        
        for f in feature_list:
            if len(f)<=1:
                self.mean.append(int(0))
                self.cov.append(int(0))
                continue
            concatenated = torch.stack(f, dim=0)
            self.mean.append(torch.mean(concatenated,dim=0).reshape(1,-1))
            self.cov.append(torch.cov(concatenated.T))
          
        del x,y,feature_list,fea,f,concatenated


        print(f"client{self.id} train_samples:{self.train_samples} cost time :{time.time() - start_time}")


        

    def train(self):

        start = time.time()
        with torch.no_grad():

            label_syn = torch.concat([torch.ones(self.num_fea, dtype=torch.long, requires_grad=False) * i for i in range(self.num_classes)]).view(-1)

            feature_real_vir = [[] for i in range(self.num_classes)]
            strat_class = self.strat_class

            for i,f in enumerate(self.feature_list):
                if i >= strat_class:
                    f.append(self.mean[i].squeeze())
                    calibrated_mean_cov = []
                    for q in f:
                        calibrated_mean_cov.append(distribution_calibration(q,self.mean[:strat_class],self.cov[:strat_class],self.m,self.global_class_num))
                    lenth = len(f)
                 
                    for l in range(lenth):  
                        v = np.random.multivariate_normal(mean=calibrated_mean_cov[l][0].ravel(), cov=calibrated_mean_cov[l][1], tol = 1e-6,size=self.num_fea,check_valid ="raise")
                        f.extend(torch.tensor(np.array(v)).to(self.device))
                else:
                    v = np.random.multivariate_normal(mean=self.mean[i].cpu().numpy().ravel(), cov=self.cov[i].cpu().numpy(), tol = 1e-5,size=self.num_fea,check_valid ="raise")
                    f.extend(torch.tensor(np.array(v)).to(self.device))
   
                self.feature_list[i] = f
      

            del self.mean,self.cov

        optimizer_my= torch.optim.SGD(self.my_model.parameters(), lr=self.fea_lr,weight_decay=1e-5,momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.my_model_personal = copy.deepcopy(self.my_model)
        optimizer_personal= torch.optim.SGD(self.my_model_personal.parameters(), lr=self.fea_lr,weight_decay=1e-5,momentum=0.9)


        self.my_model.train()
        self.my_model_personal.train()
        self.global_head.eval()

        client_class_num = read_client_json(self.dataset,self.id)[1]
        label_syn_personal = []
        plen = 0
        for i,num in enumerate(client_class_num):
            if num[1] !=0:
                plen+=1
                label_syn_personal.append(torch.ones(self.num_fea, dtype=torch.long, requires_grad=False) * i)
        label_syn_personal = torch.concat(label_syn_personal).view(-1)

        for epoch in range(self.crt_epoch):
            for i,l in enumerate(feature_real_vir):
          
                feature_real_vir[i] = random.sample(self.feature_list[i], k=self.num_fea)
            fea_train = []
            fea_train_personal = []
            for i,sub in enumerate(feature_real_vir):
                fea_train.extend(sub)
                if client_class_num[i][1] != 0:
                    fea_train_personal.extend(sub)
                    
            fea_train = torch.stack(fea_train)
            fea_train_personal = torch.stack(fea_train_personal)

            assert len(fea_train) == self.num_fea*self.num_classes
         
            assert len(fea_train_personal) == self.num_fea*plen

            loss = self.retrain_classfier(self.my_model,fea_train,label_syn,optimizer_my)
            loss_personal = self.retrain_classfier(self.my_model_personal,fea_train_personal,label_syn_personal,optimizer_personal)

            if epoch%10 == 0:
                print(epoch,loss.item())
                print(epoch,loss_personal.item())

        print(f"client{self.id} train cost time :{time.time() - start}")
        
    def retrain_classfier(self,model,fea_train, label_syn,optimizer):
        train_syn = TensorDataset(fea_train, label_syn)
        trainloader = DataLoader(dataset=train_syn,batch_size=self.cr_batch_size ,shuffle=True)
        for data_batch in trainloader:
            images, labels = data_batch
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            loss_net = self.criterion(outputs, labels)
            optimizer.zero_grad()
            loss_net.backward()
            optimizer.step()
        return loss_net
    
    def update_classfier(self,model):
        model.eval()
        head_params = model.state_dict()
        state_dict = self.model.state_dict()
        state_dict['head.bias'] = head_params['bias']
        state_dict['head.weight'] = head_params['weight']
        self.model.load_state_dict(state_dict)





