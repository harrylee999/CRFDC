import numpy as np
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, save_file,classify_label,show_clients_data_distribution
from utils.long_tailed_cifar import train_long_tail
from utils.sample_dirichlet import clients_indices
import copy
import argparse

# Allocate data to users
def generate_data(dir_path, num_clients, num_classes, niid, imb,seed,args):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, imb,seed):
        return
    
    if args.dataset == 'cifar10':
        # Get Cifar10 data
        transform = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        trainset = torchvision.datasets.CIFAR10(
            root=dir_path+"rawdata", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root=dir_path+"rawdata", train=False, download=True, transform=transform)
    elif args.dataset == 'cifar100':

        transform = transforms.Compose(
        [transforms.ToTensor(), 
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])

        trainset = torchvision.datasets.CIFAR100(
            root=dir_path+"rawdata", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(
            root=dir_path+"rawdata", train=False, download=True, transform=transform)
    elif args.dataset == 'fminist':
        transform = transforms.Compose(
        [transforms.ToTensor(), 
            transforms.Normalize((0.2860366729433025), (0.35288708155778725))
            ])

        trainset = torchvision.datasets.FashionMNIST(
            root=dir_path+"rawdata", train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(
            root=dir_path+"rawdata", train=False, download=True, transform=transform)

        
    elif args.dataset == 'cinic10':
        cinic_directory = dir_path+"rawdata"
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]



        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cinic_mean,std=cinic_std)])
        trainset = torchvision.datasets.ImageFolder(
            cinic_directory + '/train', transform=transform)
        testset = torchvision.datasets.ImageFolder(
            cinic_directory + '/test', transform=transform)

    testdata = []

    for i in range(len(testset)):
        testdata.append(testset[i])

    list_label2indices = classify_label(trainset, num_classes)
    total_class_num, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices),num_classes,imb)

   
  
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), num_classes,num_clients,niid, seed)
    original_dict_per_client = show_clients_data_distribution(trainset.targets, list_client2indices,num_classes)


   
 
    save_file(config_path, train_path, test_path, list_client2indices, trainset,testdata, num_clients, num_classes,niid,imb,seed,original_dict_per_client,total_class_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('-data', "--dataset", type=str, default="cifar10")
    parser.add_argument('-nc', "--num_clients", type=int, default=10)
    parser.add_argument('-if','--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('-noniid','--non_iid_alpha', type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)  # random and transforms

    if args.dataset == 'cinic10':
        dir_path = "data/cinic10/"
        num_classes = 10
    elif args.dataset == 'cifar10':
        dir_path = "data/cifar10/"
        num_classes = 10
    elif args.dataset == 'cifar100':
        dir_path = "data/cifar100/"
        num_classes = 100
    elif args.dataset == 'fminist':
        dir_path = "data/fminist/"
        num_classes = 10
    else:
        print("check your dataset")
        exit()


    generate_data(dir_path, args.num_clients, num_classes, args.non_iid_alpha,args.imb_factor,args.seed,args)