### Requirements

Please install the required packages. The code is compiled with Python 3.7 dependencies in a virtual environment via

```pip install -r requirements.txt```

### Instructions

Example codes to run CRFDC with FedAvg on CIFAR10 is given here.

More completed codebase will be released upon acceptance. 

#### 1. Generate Non-IID and Long-Tailed Data:
CIFAR-10 , 10 clients, imbalance_factor = 0.01, non-IID-alpha = 0.5.
```bash
cd ./dataset 
python generate_dataset.py -data cifar10 -nc 10 -if 0.01 -noniid 0.5 --seed 1
```

The output as follows:


<details>
    <summary>Show more</summary>
Original number of samples of each label:
[5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
All num_data_train
12406
client:0:  1241 [(0, 649), (1, 341), (2, 109), (3, 27), (4, 0), (5, 0), (6, 53), (7, 29), (8, 12), (9, 21)]
client:1:  1241 [(0, 545), (1, 571), (2, 125), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]
client:2:  1241 [(0, 873), (1, 0), (2, 47), (3, 11), (4, 9), (5, 155), (6, 86), (7, 24), (8, 36), (9, 0)]
client:3:  1241 [(0, 1241), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]
client:4:  1241 [(0, 769), (1, 425), (2, 47), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]
client:5:  1241 [(0, 242), (1, 615), (2, 381), (3, 3), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]
client:6:  1240 [(0, 255), (1, 283), (2, 229), (3, 42), (4, 130), (5, 59), (6, 93), (7, 86), (8, 34), (9, 29)]
client:7:  1240 [(0, 371), (1, 10), (2, 395), (3, 225), (4, 239), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]
client:8:  1240 [(0, 25), (1, 0), (2, 329), (3, 633), (4, 81), (5, 172), (6, 0), (7, 0), (8, 0), (9, 0)]
client:9:  1240 [(0, 30), (1, 752), (2, 134), (3, 136), (4, 186), (5, 1), (6, 0), (7, 0), (8, 1), (9, 0)]
Saving to disk.
Finish generating dataset.
</details>
<br/>
#### 2. Federated Learning:

Noted that CRFDC is applied after federated learning and can also be seamlessly integrated with most existing federated learning algorithms to further improve performance.

Here,   we use  FedAvg.

```bash
cd ./system
python main.py -algo FedAvg -data cifar10 -dev cuda --seed 1
```

#### 3. Perform CRFDC:

Here, we perform CRFDC based on the feature extractor of the global model trained by previous FedAvg.

```bash
cd ./system
python main.py -algo CRFDC -data cifar10 -dev cuda --seed 1 --model_path ../results/cifar10/model/FedAvgbest.pt --num_fea 50 --m 2
```

### Calculating Mutual Information

```bash
cd ./system
python get_mi.py -data cifar10 -dev cuda --seed 1 --model_path ../results/cifar10/model/FedAvgbest.pt
```



