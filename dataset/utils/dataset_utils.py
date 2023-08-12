import os
import ujson
import numpy as np
import torch


def check(config_path, train_path, test_path, num_clients, num_classes, niid, imb,seed):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_classes'] == num_classes and \
            config['non_iid_alpha'] == niid and \
            config['imb_factor'] == imb and \
            config['seed'] == seed :
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1

def show_clients_data_distribution(dataset_label, clients_indices: list, num_classes):
    dict_per_client = []

    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = dataset_label[idx]
            nums_data[label] += 1
        new_num_data = []
        total = sum(nums_data)
        for i in range(num_classes):
            new_num_data.append((i,nums_data[i]))
        dict_per_client.append((total,new_num_data))
        print(f'client:{client}:  {total} {new_num_data}')
    return dict_per_client



def save_file(config_path, train_path, test_path,train_index, train_data, test_data, num_clients, num_classes, niid, imb,seed,statistic,total):
    new_total = []
    for i,n in enumerate(total):
        new_total.append((i,n))
    # sorted_total = sorted(new_total, key=lambda x: x[1],reverse=True)


    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid_alpha': niid, 
        'imb_factor':imb, 
        'seed': seed,
        'Total class number':new_total,
        'Size of samples for labels in clients': statistic, 
        
    }

    # gc.collect()
    print("Saving to disk.\n")

    for client in range(num_clients):
        traindata = []
        for i in train_index[client]:
            traindata.append(train_data[i])
        
        torch.save(traindata,train_path + str(client) + '.pkl')

    torch.save(test_data,test_path+ 'test.pkl')

    with open(config_path, 'w') as f:
        ujson.dump(config, f,indent=1)

    print("Finish generating dataset.\n")




