import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms


class clientAVG(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        
    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                x = transform_train(x)
                self.optimizer.zero_grad()
                _,output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        print(f"client{self.id} train_samples:{self.train_samples} train cost time :{time.time() - start_time}")


