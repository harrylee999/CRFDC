import time
from flcore.clients.clientcrfdc import clientCRFDC
from flcore.servers.serverbase import Server
from tqdm import tqdm
from utils.data_utils import read_total_json,read_client_json,TensorDataset,repair_cov,get_head_class
import torch
import numpy as np
import copy

class FedCRFDC(Server):
    def __init__(self, args):
        super().__init__(args)
        self.num_fea = args.num_fea
        self.fea_dim = args.fea_dim
        self.fea_lr = args.fea_lr
        self.crt_epoch = args.crt_epoch
        self.head_ratio = args.head_ratio

        self.set_clients(args, clientCRFDC)
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
                client.cfmv()
                client_dict_per_class = read_client_json(self.dataset,client.id)[1]
        
                for i in range(self.num_classes):
                    if client_dict_per_class[i][1] <=1:
                        total_class_num[i][1]-=client_dict_per_class[i][1]
                        continue
        
                    global_mean[i] = global_mean[i] + client.mean[i]*torch.tensor(client_dict_per_class[i][1]/total_class_num[i][1]).to(self.device)
                    global_cov[i] = global_cov[i] + client.cov[i]*torch.tensor((client_dict_per_class[i][1]-1)/(total_class_num[i][1]-1)).to(self.device)\
                        + client.mean[i].T*client.mean[i]*torch.tensor(client_dict_per_class[i][1]/(total_class_num[i][1]-1)).to(self.device)

                #del client.mean,client.cov

            for i in range(self.num_classes):
                if total_class_num[i][1] != 0:
                    global_cov[i] = global_cov[i] - torch.tensor(total_class_num[i][1]/(total_class_num[i][1]-1)).to(self.device)*global_mean[i].T*global_mean[i]


        self.uploaded_weights = []#num of samples
        self.uploaded_models = []
        tot_samples = 0

        for i,cov in enumerate(global_cov):
            global_cov[i] = repair_cov(cov)
        #send gloal mean cov to clients
        client_headmodel =  torch.nn.Linear(self.fea_dim, self.num_classes).to(self.device)
        for client in self.clients:
         
            client.global_class_num = total_class_num
            client.mean = global_mean
            client.cov = global_cov
            client.strat_class = get_head_class(self.dataset,self.head_ratio)
            client.my_model = copy.deepcopy(client_headmodel)
            client.train()
            client.update_classfier(client.my_model_personal)
            self.evaluate(client.model,client.id)
    

            tot_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.my_model)
        print()
        print(f"Personal average acc:{round(sum(self.personal_acc)/len(self.personal_acc),4)}")
        


        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        self.global_head = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_head.parameters():
            param.data.zero_()
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            for server_param, client_param in zip(self.global_head.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

        head_params = self.global_head.state_dict()
        state_dict = self.global_model.state_dict()
        state_dict['head.bias'] = head_params['bias']
        state_dict['head.weight'] = head_params['weight']
        self.global_model.load_state_dict(state_dict)

        self.evaluate()

        torch.save(self.global_model.state_dict(),"../results/"+ self.dataset+"/model/"+self.algorithm+"best.pt")

        print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)
    

