import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 

root = '../path'
train_dataset = datasets.MNIST(root, train=False, transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=10000, shuffle=True, num_workers=0, drop_last=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_test(model_name):
    model = torch.load(f'./models/{model_name}.pth')
    model.to(device)
    model.eval()

    a = 0
    for i,data in enumerate(train_loader):
        
        images, labels = data
        images = torch.flatten(images, start_dim=-2)
        
        images = images.to(device)
        labels = labels.to(device)

        predict = model(images).squeeze()
        predict = torch.argmax(predict, dim=-1)
        for p, l in zip(predict, labels):
            if int(p) == int(l):
                a += 1
    auc = a / 10000
        
    print("model:",model_name, auc)
        


run_test("CrossEntropyLoss_model1")

run_test("CrossEntropyLoss_model2")

run_test("CrossEntropyLoss_model3")

# run_test("MSELoss")



