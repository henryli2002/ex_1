import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
import torch.optim as optim
from model import network # 导入模型文件中的网络模型类
import matplotlib.pyplot as plt

# 配置参数
batch_size = 64  # 每个批次的样本数量
epoch_num = 1  # 训练的总epoch数
learning_rate = 0.1  # 学习率
num_print = 100  # 每训练多少批次输出一次信息


root = '../path'  # 数据集存放的根目录

# 创建MNIST数据集
train_dataset = datasets.MNIST(root, train=True, transform=transforms.ToTensor(), download=True)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

# 检测是否有可用的GPU，如果有就使用它
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义一个函数来进行训练
def run_train(loss_fc, optimizer_fc_name, optimizer_fc, list, list_value):
    model = network()  # 否则使用普通的网络模型
    model.to(device)  # 将模型移动到GPU（如果可用）
    if (optimizer_fc_name == "Monmentum"):
        optimizer = optimizer_fc(model.parameters(), lr=learning_rate, momentum=0.99) 
    else:
        optimizer = optimizer_fc(model.parameters(), lr=learning_rate) 

    model.train()  # 将模型设置为训练模式
    for epoch in range(epoch_num):
        running_loss = 0.0
        for i, data in enumerate(train_loader, start=1):
            optimizer.zero_grad()  # 梯度清零
            images, labels = data
            images = torch.flatten(images, start_dim=-2)  # 将图像扁平化
            images = images.to(device)  # 将数据移动到GPU
            labels = labels.to(device)  # 将标签移动到GPU

            predict = model(images).squeeze()  # 前向传播，获取模型预测结果

            loss = loss_fc(predict, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()

            if i % num_print == 0:
                list.append(i)
                list_value.append(running_loss / num_print)
                running_loss = 0.0

    # 保存训练好的模型
    torch.save(model, f'./models/{optimizer_fc_name}_model.pth')

  # 使用随机梯度下降
optimizer = optim.SGD
a = []
a_value = []
run_train(nn.CrossEntropyLoss(), "SDG", optimizer, a, a_value)

# 使用Momentum优化器
optimizer = optim.SGD
b = []
b_value = []
run_train(nn.CrossEntropyLoss(), "Momentum", optimizer, b, b_value)

# 使用Adagrad优化器
optimizer = optim.Adagrad
c = []
c_value = []
run_train(nn.CrossEntropyLoss(), "Adagrad", optimizer, c, c_value)

# 使用Adam优化器
optimizer = optim.Adam
d = []
d_value = []
run_train(nn.CrossEntropyLoss(), "Adam", optimizer, d, d_value)

# 使用Adamax优化器
optimizer = optim.Adamax
e = []
e_value = []
run_train(nn.CrossEntropyLoss(), "Adamax", optimizer, e, e_value)

plt.plot(a, a_value, marker='o', linestyle='-', color='blue', label='SGD')
plt.plot(b, b_value, marker='o', linestyle='-', color='red', label='Momentum')
plt.plot(c, c_value, marker='o', linestyle='-', color='green', label='Adagrad')
plt.plot(d, d_value, marker='o', linestyle='-', color='yellow', label='Adam')
plt.plot(e, e_value, marker='o', linestyle='-', color='cyan', label='Adamax')

plt.legend()
plt.xlabel('batch_number')
plt.ylabel('loss')
plt.title('test')
plt.show()