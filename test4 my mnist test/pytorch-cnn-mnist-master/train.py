#%% 导入模块
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from model import CNN
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
#%% tensorboard
#使用指定的文件夹名创建摘要编写器。
writer = SummaryWriter('./logs/')
#%% 训练设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#%% 参数定义
EPOCH = 2
BATCH_SIZE = 128
LR = 1E-3
#%% 下载数据集
train_file = datasets.MNIST(
    root='./dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_file = datasets.MNIST(
    root='./dataset/',
    train=False,
    transform=transforms.ToTensor()
)
#%% 数据可视化
##% 训练数据可视化
train_data = train_file.data
train_targets = train_file.targets
print(train_data.size())  # [60000, 28, 28]
print(train_targets.size())  # [60000]
plt.figure(figsize=(9, 9)) #figsize 设置图形的大小，a 为图形的宽， b 为图形的高，单位为英寸
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title(train_targets[i].numpy())
    plt.axis('off') #关闭坐标轴
    plt.imshow(train_data[i], cmap='gray')#imshow的作用就是无论怎样的输入数据都映射到0-255
    # cmap='gray'绘制风格黑白色
plt.show()
##% 测试数据可视化
test_data = test_file.data
test_targets = test_file.targets
print(test_data.size())  # [10000, 28, 28]
print(test_targets.size())  # [10000]
plt.figure(figsize=(9, 9))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title(test_targets[i].numpy())
    plt.axis('off')
    plt.imshow(test_data[i], cmap='gray')
plt.show()
#%% 制作数据加载器
train_loader = DataLoader(
    dataset=train_file,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_file,
    batch_size=BATCH_SIZE,
    shuffle=False
)
#%% 模型结构
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv = nn.Sequential(
#             # [BATCH_SIZE, 1, 28, 28]
#             nn.Conv2d(1, 32, 5, 1, 2),
#             # [BATCH_SIZE, 32, 28, 28]
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             # [BATCH_SIZE, 32, 14, 14]
#             nn.Conv2d(32, 64, 5, 1, 2),
#             # [BATCH_SIZE, 64, 14, 14]
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             # [BATCH_SIZE, 64, 7, 7]
#         )
#         self.fc = nn.Linear(64 * 7 * 7, 10)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         y = self.fc(x)
#         return y
#%% 创建模型
model = CNN().to(device)
optim = torch.optim.Adam(model.parameters(), LR)
lossf = nn.CrossEntropyLoss()
#%% 定义计算整个训练集或测试集loss及acc的函数
def calc(data_loader):
    loss = 0
    total = 0
    correct = 0
    with torch.no_grad():#模型训练的时候，爆显存了，可以调整batch，对数据进行crop等等操作
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            loss += lossf(output, targets)
            correct += (output.argmax(1) == targets).sum()
            total += data.size(0)
    loss = loss.item()/len(data_loader)
    acc = correct.item()/total
    return loss, acc
#%% 训练过程打印函数
def show():
    # 定义全局变量
    if epoch == 0:
        global model_saved_list #在global 局部函数对全局变量重新赋值
        global temp
        temp = 0
    # 打印训练的EPOCH和STEP信息
    header_list = [
        f'EPOCH: {epoch+1:0>{len(str(EPOCH))}}/{EPOCH}',
        f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}'
    ] #header_list列出所有的输出      f 使字符串的代替速度更快
    header_show = ' '.join(header_list) #字符串的连接输出
    print(header_show, end=' ') #end=' '意思是末尾不换行，加空格
    # 打印训练的LOSS和ACC信息
    loss, acc = calc(train_loader)
    writer.add_scalar('loss', loss, epoch+1)
    writer.add_scalar('acc', acc, epoch+1)
    train_list = [
        f'LOSS: {loss:.4f}',
        f'ACC: {acc:.4f}'
    ]
    train_show = ' '.join(train_list)
    print(train_show, end=' ')
    # 打印测试的LOSS和ACC信息
    val_loss, val_acc = calc(test_loader)
    writer.add_scalar('val_loss', val_loss, epoch+1)
    writer.add_scalar('val_acc', val_acc, epoch+1)
    test_list = [
        f'VAL-LOSS: {val_loss:.4f}',
        f'VAL-ACC: {val_acc:.4f}'
    ]
    test_show = ' '.join(test_list)
    print(test_show, end=' ')
    # 保存最佳模型
    if val_acc > temp:
        model_saved_list = header_list+train_list+test_list
        torch.save(model.state_dict(), 'model.pt')
        temp = val_acc
#%% 训练模型
for epoch in range(EPOCH):
    start_time = time.time()
    for step, (data, targets) in enumerate(train_loader):
        optim.zero_grad()
        data = data.to(device)
        targets = targets.to(device)
        output = model(data)
        loss = lossf(output, targets)
        acc = (output.argmax(1) == targets).sum().item()/BATCH_SIZE
        loss.backward()
        optim.step()
        print(
            f'EPOCH: {epoch+1:0>{len(str(EPOCH))}}/{EPOCH}',
            f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}',
            f'LOSS: {loss.item():.4f}',
            f'ACC: {acc:.4f}',
            end='\r'
        )
    show()
    end_time = time.time()
    print(f'TOTAL-TIME: {round(end_time-start_time)}')
#%% 打印并保存最优模型的信息
model_saved_show = ' '.join(model_saved_list)
print('| BEST-MODEL | '+model_saved_show)
with open('model.txt', 'a') as f: #直接写入保存，速度快。 #a是加入文件已经存在，是需要在末尾写入即可
    f.write(model_saved_show+'\n')

'''
torch.Size([60000, 28, 28])
torch.Size([60000])
torch.Size([10000, 28, 28])
torch.Size([10000])
EPOCH: 1/2 STEP: 469/469 LOSS: 0.0643 ACC: 0.9808 VAL-LOSS: 0.0584 VAL-ACC: 0.9808 TOTAL-TIME: 171
EPOCH: 2/2 STEP: 469/469 LOSS: 0.0403 ACC: 0.9875 VAL-LOSS: 0.0378 VAL-ACC: 0.9865 TOTAL-TIME: 171
| BEST-MODEL | EPOCH: 2/2 STEP: 469/469 LOSS: 0.0403 ACC: 0.9875 VAL-LOSS: 0.0378 VAL-ACC: 0.9865

'''
