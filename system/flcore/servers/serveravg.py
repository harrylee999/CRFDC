import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from tqdm import tqdm
import torch
import os
class FedAvg(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args, clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in tqdm(range(1, self.global_rounds+1), desc='server-training'):
     
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            print()
            for client in self.selected_clients:
                client.train()
            self.receive_models()
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
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                torch.save(self.global_model.state_dict(),result_path+self.algorithm+"best.pt")
            



