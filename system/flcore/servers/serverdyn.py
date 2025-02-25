import copy
import torch
from flcore.clients.clientdyn import clientDyn
from flcore.servers.serverbase import Server
import time
from tqdm import tqdm
import os

class FedDyn(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args,clientDyn)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

  
        self.alpha = args.alpha
        
        # self.server_state = copy.deepcopy(args.model)
        # # for param in self.server_state.parameters():
        # #     param.data = torch.zeros_like(param.data)
        # # 将所有参数值设置为0.0
        # for param_tensor in self.server_state.state_dict():
        #     self.server_state.state_dict()[param_tensor].fill_(float(0.0))

        self.h = {
        key: torch.zeros(params.shape, device=args.device)
        for key, params in args.model.state_dict().items()
    }

   



    def train(self):
        for i in tqdm(range(1, self.global_rounds+1), desc='server-training'):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()


            for client in self.selected_clients:
                client.train()


            self.receive_models()
    
            # self.update_server_state()
            self.aggregate_parameters()
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")
            self.evaluate()
            print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)

            if i%self.save_gap == 0:
                self.save_results(i)
            if self.current_acc < self.test_acc[-1]:
                self.current_acc = self.test_acc[-1]
                result_path = "../results/"+ self.dataset+"/model/"
                result_path_per = "../results/"+ self.dataset+"/model/"+self.algorithm+"-per/"
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                if not os.path.exists(result_path_per):
                    os.makedirs(result_path_per)

                torch.save(self.global_model.state_dict(),result_path+self.algorithm+"best.pt")
                for client in self.clients:
                    torch.save(client.model.state_dict(),result_path_per+str(client.id)+".pt")
                   



    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)


        self.h = {
            key: prev_h
            - self.alpha * 1. /self.num_clients * sum(theta[key] - old_params for theta in self.uploaded_models)
            for (key, prev_h), old_params in zip(self.h.items(), self.global_model.state_dict().values())
        }

        # temp_h = {
        #     sum(theta[key] for theta in self.uploaded_models)
        #     for (key, prev_h) in self.h.items()
        # }

        # self.h = {
        #     key: prev_h
        #     - self.alpha * 1. /self.num_clients * (temp_h[key] - old_params)
        #     for (key, prev_h), old_params in zip(self.h.items(), self.global_model.state_dict().values())
        # }


        new_parameters = {
            key: (1. / self.num_join_clients) * sum(theta[key] for theta in self.uploaded_models)
            for key in self.global_model.state_dict().keys()
        }
        new_parameters = {
            key: params - (1. / self.alpha) * h_params
            for (key, params), h_params in zip(new_parameters.items(), self.h.values())
        }
        self.global_model.load_state_dict(new_parameters)

        # fedavg_global_params = self.global_model.state_dict()
 

        # for name_param in self.uploaded_models[0]:
        #     list_values_param = []
        #     for dict_local_params in self.uploaded_models:
        #         list_values_param.append(dict_local_params[name_param] / self.num_join_clients)
        #     value_global_param = sum(list_values_param)
        #     fedavg_global_params[name_param] = value_global_param
        # self.global_model.load_state_dict(fedavg_global_params)

        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        # for param in self.global_model.parameters():
        #     param.data = torch.zeros_like(param.data)
            
        # for client_model in self.uploaded_models:
        #     self.add_parameters(client_model)
        # state_param = self.server_state.state_dict()
        # server_param = self.global_model.state_dict()
        # for name_param in self.uploaded_models[0]:
        #     a = server_param[name_param] - (1/self.alpha) * state_param[name_param]
        #     server_param[name_param] = a
        
        # self.global_model.load_state_dict(server_param)

        # for server_param, state_param in zip(self.global_model.parameters(), self.server_state.parameters()):
        #     server_param.data -= (1/self.alpha) * state_param

    def update_server_state(self):
        assert (len(self.uploaded_models) > 0)
        model_delta = copy.deepcopy(self.global_model)
        fedavg_global_params = copy.deepcopy(self.global_model.state_dict())
        for name_param in self.uploaded_models[0]:
            list_values_param = []
            for dict_local_params in self.uploaded_models:
                list_values_param.append((dict_local_params[name_param] - fedavg_global_params[name_param]) / self.num_join_clients)
            value_global_param = sum(list_values_param)
            fedavg_global_params[name_param] = value_global_param
        model_delta.load_state_dict(fedavg_global_params)




        
        # for param in model_delta.parameters():
        #     param.data = torch.zeros_like(param.data)

        # for client_model in self.uploaded_models:
        #     for server_param, client_param, delta_param in zip(self.global_model.parameters(), client_model.parameters(), model_delta.parameters()):
        #         delta_param.data += (client_param - server_param) / self.num_clients
        # state_param = self.server_state.state_dict()
        # delta_param = model_delta.state_dict()
        # for name_param in self.uploaded_models[0]:
        #     # print(state_param[name_param])
        #     # print(state_param[name_param] - self.alpha * delta_param[name_param])
        #     a = state_param[name_param] - self.alpha * delta_param[name_param]
        #     state_param[name_param] = a
        
        # self.server_state.load_state_dict(state_param)
        for state_param, delta_param in zip(self.server_state.parameters(), model_delta.parameters()):
            state_param.data -= self.alpha * delta_param