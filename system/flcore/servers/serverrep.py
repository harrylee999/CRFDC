import random
from flcore.clients.clientrep import clientRep
from flcore.servers.serverbase import Server
from threading import Thread
import time
import copy
from tqdm import tqdm
import torch
import os


class FedRep(Server):
    def __init__(self, args):
        super().__init__(args)

        # select slow clients
        self.set_clients(args, clientRep)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
    


    def train(self):
        for i in tqdm(range(1, self.global_rounds+1), desc='server-training'):
     
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()


            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()


            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate personal model")


            self.personal_acc = []
            for client in self.clients:
          
                self.evaluate(client.pmodel,client.id)

            self.test_acc.append(round(sum(self.personal_acc)/len(self.personal_acc),4))
            print(f"Personal average acc:{self.test_acc}")
            print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)

            if i%self.save_gap == 0:
                self.save_results(i)

            if self.current_acc < self.test_acc[-1]:
                self.current_acc = self.test_acc[-1]
                result_path = "../results/"+ self.dataset+"/model/"+self.algorithm+"-per/"
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                for client in self.clients:
                    torch.save(client.pmodel.state_dict(),result_path+str(client.id)+".pt")
                    
    def get_acc(self):
        result_path_per = "../results/"+ self.dataset+"/model/"+self.algorithm+"-per/"
        self.uploaded_models = []
        for client in self.clients:
            client.model.load_state_dict(torch.load(result_path_per+str(client.id)+".pt"))
            client.fine_tune()
            self.evaluate(client.model,client.id)
            self.uploaded_models.append(client.model.state_dict())
        acc = round(sum(self.personal_acc)/len(self.personal_acc),4)
        print(f"Personal average acc:{acc}")
        global_params = self.global_model.state_dict()
        for name_param in self.uploaded_models[0]:
            list_values_param = []
            for dict_local_params in self.uploaded_models:
                list_values_param.append(dict_local_params[name_param] * 1/len(self.uploaded_models))
            value_global_param = sum(list_values_param)
            global_params[name_param] = value_global_param
        self.global_model.load_state_dict(global_params)   
        self.evaluate()
        torch.save(self.global_model.state_dict(),"../results/"+ self.dataset+"/model/"+self.algorithm+"best.pt")

    def receive_models(self):
        assert (len(self.selected_clients) > 0)



        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.selected_clients:

            tot_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.base.state_dict())
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        fedavg_global_params = self.global_model.state_dict()
        for name_param in self.uploaded_models[0]:
            list_values_param = []
            for dict_local_params, local_weight in zip(self.uploaded_models, self.uploaded_weights):
                list_values_param.append(dict_local_params[name_param] * local_weight)
            value_global_param = sum(list_values_param)
            fedavg_global_params['base.'+name_param] = value_global_param
        self.global_model.load_state_dict(fedavg_global_params)