
from utils import simplebinmi
from utils.data_utils import read_data
from torch.utils.data import DataLoader
from flcore.trainmodel.resnet import resnet18,resnet8,BaseHeadSplit
import torch
import copy
import numpy as np
import random
import argparse


def get_acc(model):
    model.eval()
    with torch.no_grad():
        num_corrects = {i: 0 for i in range(10)}
        total = {i: 0 for i in range(10)}
        total_corrects = 0
        total_samples = 0
        for data_batch in test_loader:
            images, labels = data_batch
            images, labels = images.to(device), labels.to(device)
            _, outputs = model(images)
            _, predicts = torch.max(outputs, -1)
            for i in range(len(labels)):
                total[labels[i].item()] += 1
                total_samples += 1
                if predicts[i] == labels[i]:
                    num_corrects[labels[i].item()] += 1
                    total_corrects += 1
            accuracy = []
            for i in range(10):
                accuracy_i = num_corrects[i] / total[i]
                accuracy.append(accuracy_i)
        total_accuracy = total_corrects / total_samples
    
        print(total_accuracy,accuracy)

def get_mi(model):
    model.eval()
    indices = list(range(0, 10000))
    target_indices = random.sample(indices, k=2000)
    target_data = [data_test[i] for i in target_indices]
    data = torch.tensor([]).to(device)
    label = []
    for d,l in target_data:
        d = d.to(device).unsqueeze(0)
        data = torch.cat((data,d))
        label.append(l)
    label = np.array(label)
    running_mi_ty = []
    _,output = model(data)
    activity = output.cpu().detach().numpy()
    binxm, binym = simplebinmi.MI_cal(label, activity,NUM_TEST_MASK=len(activity))
    running_mi_ty.append(binym)
    print(f"XT:{binxm} TY:  {binym}")
    return running_mi_ty

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-dev', "--device", type=str, default="cuda:0",choices=["cpu", "cuda"])
    parser.add_argument('-data', "--dataset", type=str, default="cifar10")
    parser.add_argument("--model_path", type=str, default="../results/cifar10/model/FedAvgbest.pt")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    device= args.device
    seed = args.seed
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn 
    torch.backends.cudnn.benchmark = False

    data_test = read_data(args.dataset,idx=None,is_train=False)
    test_loader = DataLoader(data_test, 500,pin_memory=True,num_workers=4)

    model = resnet8(num_classes=10).to(device)
    head = copy.deepcopy(model.classifier)
    model.classifier = torch.nn.Identity()
    model = BaseHeadSplit(model, head)

    model_path = args.model_path
    global_params = torch.load(model_path)
    model.load_state_dict(global_params)

    get_mi(model)
    get_acc(model)