from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from tqdm import tqdm
import copy
import torch

class Local(Server):
    def __init__(self, args):
        super().__init__(args)


        self.set_clients(args, clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        self.fine_tuning_steps = args.fine_tuning_steps
        self.model_path = args.model_path



    def train(self):
        personal_acc = []
        for i in tqdm(range(1, 25+1), desc='Local-training'):
            for client in self.clients:
                client.train()
                self.evaluate(client.model,client.id)

            acc = round(sum(self.personal_acc)/len(self.personal_acc),4)
            self.personal_acc = []
            personal_acc.append(acc)
            print(f"Personal average acc:{personal_acc}")
            if i%self.save_gap == 0:
                self.test_acc = copy.deepcopy(personal_acc)
                self.save_results(i,save_model = False)


    def get_acc(self):
        personal_acc = []
        global_params = torch.load(self.model_path)
        self.global_model.load_state_dict(global_params)
        self.evaluate()

        for i in tqdm(range(1, self.fine_tuning_steps+1), desc='Local-training'):
            
            for client in self.clients:
                client.model.load_state_dict(self.global_model.state_dict())
                client.train()
                self.evaluate(client.model,client.id)

            acc = round(sum(self.personal_acc)/len(self.personal_acc),4)
            self.personal_acc = []
            personal_acc.append(acc)
            print(f"Personal average acc:{personal_acc}")
