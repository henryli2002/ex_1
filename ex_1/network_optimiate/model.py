from torch import nn

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()  # 第一句话，调用父类的构造函数
        self.layer1 = nn.Linear(784,100)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(100,10)
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        return x