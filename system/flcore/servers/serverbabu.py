import random
from flcore.clients.clientbabu import clientBABU
from flcore.servers.serverbase import Server
from tqdm import tqdm
import torch
import copy
import os

class FedBABU(Server):
    def __init__(self, args):
        super().__init__(args)
        self.model_path = args.model_path

        self.set_clients(args, clientBABU)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()


    def train(self):
        for i in tqdm(range(1, self.global_rounds+1), desc='server-training'):
            self.selected_clients = self.select_clients()
            self.send_models()


            for client in self.selected_clients:
                client.train()

            self.receive_models()

            self.aggregate_parameters()



            self.evaluate()
            if self.current_acc < self.test_acc[-1]:
                self.current_acc = self.test_acc[-1]
                torch.save(self.global_model.state_dict(),"../results/"+ self.dataset+"/model/"+self.algorithm+"best.pt")

    def get_acc(self):


        result_path = "../results/"+ self.dataset+"/model/"
        self.global_model.load_state_dict(torch.load(result_path+self.algorithm+"best.pt"))
        self.evaluate()
        for client in self.clients:
            client.model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))
            client.fine_tune()
            self.evaluate(client.model,client.id)
                            
            


        #     tot_samples += client.train_samples
        #     self.uploaded_weights.append(client.train_samples)
        #     self.uploaded_models.append(client.head_model)

        # for i, w in enumerate(self.uploaded_weights):
        #     self.uploaded_weights[i] = w / tot_samples

        # self.global_head = copy.deepcopy(self.uploaded_models[0])
        # for param in self.global_head.parameters():
        #     param.data.zero_()
        # for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
        #     for server_param, client_param in zip(self.global_head.parameters(), client_model.parameters()):
        #         server_param.data += client_param.data.clone() * w

        # self.global_model.load_state_dict(torch.load("../results/"+ self.dataset+"/model/"+self.algorithm+"best.pt"))
        # self.evaluate()
        # head_params = self.global_head.state_dict()
        # state_dict = self.global_model.state_dict()
        # state_dict['head.bias'] = head_params['bias']
        # state_dict['head.weight'] = head_params['weight']
        # self.global_model.load_state_dict(state_dict)

        # self.evaluate()


        print(f"Personal average acc:{round(sum(self.personal_acc)/len(self.personal_acc),4)}")
       


        

    


    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        receive_clients = self.selected_clients
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in receive_clients:
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