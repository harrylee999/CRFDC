import copy
import torch
import argparse
import warnings
import numpy as np
import logging

import random
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverdyn import FedDyn
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverrod import FedROD
from flcore.servers.servercreff import FedCreff
from flcore.servers.serverala import FedALA
from flcore.servers.serverrep import FedRep
from flcore.servers.servercrfdc import FedCRFDC
from flcore.servers.serverccvr import FedCCVR
from flcore.servers.serverlocal import Local
from flcore.trainmodel.resnet import resnet18,resnet8,BaseHeadSplit
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")


def run(args):
    reporter = MemReporter()
    print("Creating server and clients ...")
    if args.dataset == 'cinic10':
        args.model = resnet8(num_classes=args.num_classes).to(args.device)
    elif  args.dataset == 'cifar10':
        args.model = resnet8(num_classes=args.num_classes).to(args.device)
    elif args.dataset == 'cifar100':
        args.model = resnet8(num_classes=args.num_classes).to(args.device)
    else:
        print("check args.dataset !!!!!!")
        exit()

        # select algorithm
    if args.algorithm == "FedAvg":
        args.head = copy.deepcopy(args.model.classifier)
        args.model.classifier = torch.nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedAvg(args)

    elif args.algorithm == "FedProx":
        args.head = copy.deepcopy(args.model.classifier)
        args.model.classifier = torch.nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedProx(args)

    elif args.algorithm == "FedDyn":
        args.head = copy.deepcopy(args.model.classifier)
        args.model.classifier = torch.nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedDyn(args)   

    elif args.algorithm == "Creff":
        args.head = copy.deepcopy(args.model.classifier)
        args.model.classifier = torch.nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedCreff(args)

    elif args.algorithm == "FedROD":
        args.head = copy.deepcopy(args.model.classifier)
        args.model.classifier = torch.nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
      
        server = FedROD(args)

    elif args.algorithm == "FedRep":
        args.head = copy.deepcopy(args.model.classifier)
        args.model.classifier = torch.nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
      
        server = FedRep(args)

    elif args.algorithm == "FedBABU":
        args.head = copy.deepcopy(args.model.classifier)

        args.model.classifier = torch.nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedBABU(args)

    elif args.algorithm == "FedALA":
            server = FedALA(args)
  
    elif args.algorithm == "CRFDC":
        args.head = copy.deepcopy(args.model.classifier)
        args.model.classifier = torch.nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        global_params = torch.load(args.model_path)
        if "ALA" in args.model_path:
            new_state_dict = {}
            for key, value in global_params.items():
                if 'classifier' not in key:
                    new_key = "base."+key
                else:
                    new_key = key.replace('classifier','head')
                new_state_dict[new_key] = value
            args.model.load_state_dict(new_state_dict)
        else:
            args.model.load_state_dict(global_params)
      
        server = FedCRFDC(args)

    elif args.algorithm == "CCVR":
        args.head = copy.deepcopy(args.model.classifier)
        args.model.classifier = torch.nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        global_params = torch.load(args.model_path)
        args.model.load_state_dict(global_params)
        server = FedCCVR(args)

    elif args.algorithm == "Local":
        args.head = copy.deepcopy(args.model.classifier)
        args.model.classifier = torch.nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = Local(args)
    else:
        raise NotImplementedError
    
    if args.eval:
        server.get_acc()
    else:
        server.train()
    print("All done!")

    reporter.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Universal setting
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-dev', "--device", type=str, default="cuda:0",choices=["cpu", "cuda"])
    parser.add_argument('-data', "--dataset", type=str, default="cifar10")
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.1)
    parser.add_argument('-le', "--local_epochs", type=int, default=5)
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.4,help="Ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,help="Total number of clients")
    parser.add_argument("--save_gap", type=int, default=25,help="Rounds gap to save result")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--eval', type=bool, default=0)

    #CRFDC 
    parser.add_argument("--head_ratio", type=float, default=0.95)
    parser.add_argument("--model_path", type=str, default="../results/cifar10/model/FedAvgbest.pt")
    parser.add_argument("--cr_batch_size", type=int, default=64)
    parser.add_argument("--num_fea", type=int, default=50,help = "the number of features each Statistic to generate ")
    parser.add_argument("--fea_lr", type=float, default=0.01)
    parser.add_argument("--crt_epoch", type=int, default=30,help = "retrain head ")
    parser.add_argument("--m", type=int, default=2,help = 'the number of selected head class statistics m.')

    #FedProx 
    parser.add_argument("--mu", type=float, default=0.01)
    #FedDyn
    parser.add_argument('-al', "--alpha", type=float, default=0.01)


    #Creff
    parser.add_argument("--lr_feature", type=float, default=0.1)
    parser.add_argument('--match_epoch', type=int, default=100)

    #babu
    parser.add_argument('--fine_tuning_steps', type=int, default=5)

    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=1,
                        help="More fine-graind than its original paper.")

    args = parser.parse_args()
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn 
    torch.backends.cudnn.benchmark = False
    if args.dataset == 'cinic10':
        args.num_classes = 10
        args.model = 'resnet8'
        args.fea_dim = 128
    elif args.dataset == 'cifar10':
        args.num_classes = 10
        args.model = 'resnet8'
        args.fea_dim = 128
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.model = 'resnet8'
        args.fea_dim = 128
    else:
        print("check your dataset!!!!!!!!!!!!!")
        exit()


    print("=" * 50)
    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("=" * 50)

    run(args)


