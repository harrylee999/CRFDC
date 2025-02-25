import time
from flcore.clients.clientala import clientALA
from flcore.servers.serverbase import Server
from tqdm import tqdm
import os
import torch
import copy

class FedALA(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args,clientALA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")


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
            print("\nEvaluate global model")
            self.evaluate()
            self.personal_acc = []
            for client in self.clients:
                
                self.evaluate(copy.deepcopy(client.model),client.id)

            pacc= round(sum(self.personal_acc)/len(self.personal_acc),4)

            self.test_acc[-1] = (self.test_acc[-1],pacc)

            print(f"{i}: gloabal:{self.test_acc[-1][0]} personal:{pacc}")

            
            print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)

            if i%self.save_gap == 0:
                self.save_results(i)
                
            if self.current_acc < self.test_acc[-1][0]:
                self.current_acc = self.test_acc[-1][0]
                result_path = "../results/"+ self.dataset+"/model/"
                result_path_per = "../results/"+ self.dataset+"/model/"+self.algorithm+"-per/"
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                if not os.path.exists(result_path_per):
                    os.makedirs(result_path_per)

                torch.save(self.global_model.state_dict(),result_path+self.algorithm+"best.pt")
                for client in self.clients:
                    torch.save(client.model.state_dict(),result_path_per+str(client.id)+".pt")



    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.local_initialization(self.global_model)