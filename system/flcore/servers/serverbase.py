import torch
import os
import numpy as np
import csv
import copy
from utils.data_utils import read_data,read_client_json
from torch.utils.data import DataLoader

class Server(object):
    def __init__(self, args):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm


        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.test_acc = []
        self.personal_acc = []
 
        self.current_acc = 0

        self.save_gap = args.save_gap

        self.data_test = read_data(self.dataset,idx=None,is_train=False)
        self.test_loader = DataLoader(self.data_test, 1000,pin_memory=True,num_workers=4)


    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_samples = read_client_json(self.dataset, i)[0]
            client = clientObj(args, 
                            id=i, 
                            train_samples=train_samples)
            self.clients.append(client)


    def select_clients(self):
        selected_clients = list(np.random.choice(self.clients, self.num_join_clients, replace=False))
        
        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            client.set_parameters(self.global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        receive_clients = self.selected_clients
        self.uploaded_ids = []
        self.uploaded_weights = []#num of samples
        self.uploaded_models = []
        tot_samples = 0
        for client in receive_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.state_dict())
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
            fedavg_global_params[name_param] = value_global_param
        self.global_model.load_state_dict(fedavg_global_params)

    def save_results(self,r,save_model = True):
        algo = self.algorithm
        result_path = "../results/"+ self.dataset+"/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.test_acc)):
            algo = algo + "_test"
            file_path = result_path + "{}.csv".format(algo)
            print("File path: " + file_path)
            my_list_2d = [[x] for x in self.test_acc]
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(my_list_2d)  
        if save_model:
            model_path = os.path.join(result_path,"model")
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, self.algorithm + "_server" +str(r)+ ".pt")
            torch.save(self.global_model.state_dict(), model_path)

    # evaluate 
    def evaluate(self, client_model = None,id = None): 
        if client_model is None:
            model = self.global_model
        else:
            model = client_model
            client_each_class_num = read_client_json(self.dataset, id)[1]
            personal_class_acc = []
        model.eval()
        with torch.no_grad():
            num_corrects = {i: 0 for i in range(self.num_classes)}
            total = {i: 0 for i in range(self.num_classes)}
            total_corrects = 0
            total_samples = 0
            for data_batch in self.test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = model(images)
                _, predicts = torch.max(outputs, -1)
                for i in range(len(labels)):
                    total[labels[i].item()] += 1
                    total_samples += 1
                    if predicts[i] == labels[i]:
                        num_corrects[labels[i].item()] += 1
                        total_corrects += 1
            if client_model is not None:
                accuracy = []
                for i in range(self.num_classes):
                    accuracy_i = num_corrects[i] / total[i]
                    accuracy.append(accuracy_i)
                    if client_model is not None:
                        if client_each_class_num[i][1]!=0:
                            personal_class_acc.append(accuracy_i)
        # print()
        if client_model is not None:               
            self.personal_acc.append(round(sum(personal_class_acc)/len(personal_class_acc),4))
            # print(f"Personal acc:{self.personal_acc}")

        if client_model is None:
            total_accuracy = total_corrects / total_samples
            self.test_acc.append(total_accuracy)

        if client_model is None:
            print(f"Global acc:{self.test_acc}")




