import time
from flcore.clients.clientccvr import clientCCVR
from flcore.servers.serverbase import Server
from tqdm import tqdm
from utils.data_utils import read_total_json,read_client_json,TensorDataset,repair_cov
import torch
import numpy as np

class FedCCVR(Server):
    def __init__(self, args):
        super().__init__(args)
        self.num_fea = args.num_fea
        self.fea_dim = args.fea_dim
        self.fea_lr = args.fea_lr
        self.crt_epoch = args.crt_epoch
        self.cr_batch_size = args.cr_batch_size
        # select slow clients
        self.set_clients(args, clientCCVR)
        print("Finished creating server and clients.")

 
    def train(self):

        with torch.no_grad():
            s_t = time.time()

            self.send_models()

            print()
            global_mean = [torch.zeros(1).to(self.device) for _ in range(self.num_classes)]
            global_cov = [torch.zeros(1).to(self.device) for _ in range(self.num_classes)]
            total_class_num = read_total_json(self.dataset)

            for client in self.clients:
                client.train()
                client_dict_per_class = read_client_json(self.dataset,client.id)[1]
        
                for i in range(self.num_classes):
                    if client_dict_per_class[i][1] <=1:
                        total_class_num[i][1]-=client_dict_per_class[i][1]
                        continue
        
                    global_mean[i] = global_mean[i] + client.mean[i]*torch.tensor(client_dict_per_class[i][1]/total_class_num[i][1]).to(self.device)
                    global_cov[i] = global_cov[i] + client.cov[i]*torch.tensor((client_dict_per_class[i][1]-1)/(total_class_num[i][1]-1)).to(self.device)\
                        + client.mean[i].T*client.mean[i]*torch.tensor(client_dict_per_class[i][1]/(total_class_num[i][1]-1)).to(self.device)

                del client.mean,client.cov

            for i in range(self.num_classes):
                if total_class_num[i][1] != 0:
                    global_cov[i] = global_cov[i] - torch.tensor(total_class_num[i][1]/(total_class_num[i][1]-1)).to(self.device)*global_mean[i].T*global_mean[i]




            feature_syn = []
            start = time.time()
            for i in range(self.num_classes):
                if total_class_num[i] != 0:
                    feature_syn.extend(np.random.multivariate_normal(mean=global_mean[i].cpu().numpy().ravel(), cov=global_cov[i].cpu().numpy(), size=self.num_fea))
            print(f"generate fea num:{len(feature_syn)} time:{time.time()-start}")

            del global_mean,global_cov

            # relu = torch.nn.ReLU()
            # for i,f in enumerate(feature_syn):
            #     x = relu(torch.tensor(f))


            #     # feature_syn[i] = x.cpu().numpy()

        start = time.time()
        feature_syn = torch.tensor(np.array(feature_syn))
        label_syn = torch.concat([torch.ones(self.num_fea, dtype=torch.long, requires_grad=False) * i for i in range(self.num_classes)]).view(-1)
        
        train_syn = TensorDataset(feature_syn, label_syn)
        ccvr_model = torch.nn.Linear(self.fea_dim, self.num_classes).to(self.device)
        optimizer_ccvr = torch.optim.SGD(ccvr_model.parameters(), lr=self.fea_lr,weight_decay=1e-5,momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        ccvr_model.train()
        for epoch in range(self.crt_epoch):
            trainloader = torch.utils.data.dataloader.DataLoader(dataset=train_syn,batch_size=self.cr_batch_size,shuffle=True)
            for data_batch in trainloader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = ccvr_model(images)
                loss_net = criterion(outputs, labels)
                optimizer_ccvr.zero_grad()
                loss_net.backward()
                optimizer_ccvr.step()
            if (epoch+1)%10 == 0:
                print(epoch+1,loss_net.item())
        print("train head cost time:",time.time()-start)
        ccvr_model.eval()
        self.evaluate()
        head_params = ccvr_model.state_dict()

        state_dict = self.global_model.state_dict()
        state_dict['head.bias'] = head_params['bias']
        state_dict['head.weight'] = head_params['weight']

        self.global_model.load_state_dict(state_dict)

        self.evaluate()
        torch.save(self.global_model.state_dict(),"../results/"+ self.dataset+"/model/"+self.algorithm+"best.pt")

        print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)
    