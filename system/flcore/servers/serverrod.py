from flcore.clients.clientrod import clientROD
from flcore.servers.serverbase import Server
from tqdm import tqdm
import torch
import os
import time
from utils.data_utils import read_client_json


class FedROD(Server):
    def __init__(self, args):
        super().__init__(args)


        self.set_clients(args, clientROD)

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
            print("\nEvaluate models")
            self.evaluate()

            print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)

         




            self.personal_acc = []
            for client in self.clients:
                
                self.test_metrics(client.model,client.head,client.id)

            pacc= round(sum(self.personal_acc)/len(self.personal_acc),4)

            self.test_acc[-1] = (self.test_acc[-1],pacc)

            print(f"{i}: gloabal:{self.test_acc[-1][0]} personal:{pacc}")

            if i%self.save_gap == 0:
                self.save_results(i)

            if self.current_acc < self.test_acc[-1][0]:
                self.current_acc = self.test_acc[-1][0]
                result_path = "../results/"+ self.dataset+"/model/"
                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                torch.save(self.global_model.state_dict(),result_path+self.algorithm+"best.pt")

        # self.personal_acc = []
        # for client in self.clients:
        #     client.model.load_state_dict(torch.load("../results/"+ self.dataset+"/model/"+self.algorithm+"best.pt"))
        #     client.train()
        #     self.test_metrics(client.model,client.head,client.id)
        #     print(self.personal_acc)
   

        # print(f"Personal average acc:{round(sum(self.personal_acc)/len(self.personal_acc),4)}")




    def test_metrics(self, model,head,id):
        testloader = self.test_loader

        model.eval()

        
        num_corrects = {i: 0 for i in range(self.num_classes)}
        total = {i: 0 for i in range(self.num_classes)}
        client_each_class_num = read_client_json(self.dataset, id)[1]
        personal_class_acc = []
     

        
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                _,rep = model.base(x)
                out_g = model.head(rep)
                out_p = head(rep.detach())
                output = out_g.detach() + out_p

                _, predicts = torch.max(output, -1)
                for i in range(len(y)):
                    total[y[i].item()] += 1
                    if predicts[i] == y[i]:
                        num_corrects[y[i].item()] += 1

            accuracy = []
            for i in range(self.num_classes):
                accuracy_i = num_corrects[i] / total[i]
                accuracy.append(accuracy_i)
                if client_each_class_num[i][1]!=0:
                    personal_class_acc.append(accuracy_i)

            self.personal_acc.append(round(sum(personal_class_acc)/len(personal_class_acc),4))
            # print(f"Personal acc:{self.personal_acc}")

