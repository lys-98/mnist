import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

lr = 0.01  # 学习率
momentum = 0.5
log_interval = 10  # 跑多少次batch进行一次日志记录
epochs = 10
batch_size = 64
test_batch_size = 1000


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x  # F.softmax(x, dim=1)


def train(epoch):  # 定义每个epoch的训练细节
    model.train()  # 设置为trainning模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        optimizer.zero_grad()  # 优化器梯度初始化为零
        output = model(data)  # 把数据输入网络并得到输出，即进行前向传播
        loss = F.cross_entropy(output, target)  # 交叉熵损失函数
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 结束一次前传+反传之后，更新参数
        if batch_idx % log_interval == 0:  # 准备打印相关信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(target)  # 计算前要把变量变成Variable形式，因为这样子才有梯度

        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss 把所有loss值进行累加
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加

    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 启用GPU

    train_loader = torch.utils.data.DataLoader(  # 加载训练数据
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(  # 加载训练数据，详细用法参考我的Pytorch打怪路（一）系列-（1）
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
        ])),
        batch_size=test_batch_size, shuffle=True)

    model = LeNet()  # 实例化一个网络对象
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # 初始化优化器

    for epoch in range(1, epochs + 1):  # 以epoch为单位进行循环
        train(epoch)
        test()

    torch.save(model, 'model.pth')  # 保存模型

'''
Train Epoch: 10 [0/60000 (0%)]	Loss: 0.062415
Train Epoch: 10 [640/60000 (1%)]	Loss: 0.040618
Train Epoch: 10 [1280/60000 (2%)]	Loss: 0.008981
Train Epoch: 10 [1920/60000 (3%)]	Loss: 0.039145
Train Epoch: 10 [2560/60000 (4%)]	Loss: 0.015433
Train Epoch: 10 [3200/60000 (5%)]	Loss: 0.043156
Train Epoch: 10 [3840/60000 (6%)]	Loss: 0.013645
Train Epoch: 10 [4480/60000 (7%)]	Loss: 0.005156
Train Epoch: 10 [5120/60000 (9%)]	Loss: 0.054910
Train Epoch: 10 [5760/60000 (10%)]	Loss: 0.011806
Train Epoch: 10 [6400/60000 (11%)]	Loss: 0.023770
Train Epoch: 10 [7040/60000 (12%)]	Loss: 0.014078
Train Epoch: 10 [7680/60000 (13%)]	Loss: 0.118154
Train Epoch: 10 [8320/60000 (14%)]	Loss: 0.008374
Train Epoch: 10 [8960/60000 (15%)]	Loss: 0.003568
Train Epoch: 10 [9600/60000 (16%)]	Loss: 0.042900
Train Epoch: 10 [10240/60000 (17%)]	Loss: 0.004169
Train Epoch: 10 [10880/60000 (18%)]	Loss: 0.006452
Train Epoch: 10 [11520/60000 (19%)]	Loss: 0.012460
Train Epoch: 10 [12160/60000 (20%)]	Loss: 0.006232
Train Epoch: 10 [12800/60000 (21%)]	Loss: 0.004522
Train Epoch: 10 [13440/60000 (22%)]	Loss: 0.009550
Train Epoch: 10 [14080/60000 (23%)]	Loss: 0.005685
Train Epoch: 10 [14720/60000 (25%)]	Loss: 0.042629
Train Epoch: 10 [15360/60000 (26%)]	Loss: 0.063301
Train Epoch: 10 [16000/60000 (27%)]	Loss: 0.025343
Train Epoch: 10 [16640/60000 (28%)]	Loss: 0.058163
Train Epoch: 10 [17280/60000 (29%)]	Loss: 0.031539
Train Epoch: 10 [17920/60000 (30%)]	Loss: 0.040952
Train Epoch: 10 [18560/60000 (31%)]	Loss: 0.010704
Train Epoch: 10 [19200/60000 (32%)]	Loss: 0.044232
Train Epoch: 10 [19840/60000 (33%)]	Loss: 0.031729
Train Epoch: 10 [20480/60000 (34%)]	Loss: 0.027116
Train Epoch: 10 [21120/60000 (35%)]	Loss: 0.004112
Train Epoch: 10 [21760/60000 (36%)]	Loss: 0.018198
Train Epoch: 10 [22400/60000 (37%)]	Loss: 0.029132
Train Epoch: 10 [23040/60000 (38%)]	Loss: 0.024418
Train Epoch: 10 [23680/60000 (39%)]	Loss: 0.014367
Train Epoch: 10 [24320/60000 (41%)]	Loss: 0.039588
Train Epoch: 10 [24960/60000 (42%)]	Loss: 0.019524
Train Epoch: 10 [25600/60000 (43%)]	Loss: 0.014653
Train Epoch: 10 [26240/60000 (44%)]	Loss: 0.011909
Train Epoch: 10 [26880/60000 (45%)]	Loss: 0.141710
Train Epoch: 10 [27520/60000 (46%)]	Loss: 0.023481
Train Epoch: 10 [28160/60000 (47%)]	Loss: 0.009125
Train Epoch: 10 [28800/60000 (48%)]	Loss: 0.020804
Train Epoch: 10 [29440/60000 (49%)]	Loss: 0.105852
Train Epoch: 10 [30080/60000 (50%)]	Loss: 0.030783
Train Epoch: 10 [30720/60000 (51%)]	Loss: 0.015300
Train Epoch: 10 [31360/60000 (52%)]	Loss: 0.003109
Train Epoch: 10 [32000/60000 (53%)]	Loss: 0.026351
Train Epoch: 10 [32640/60000 (54%)]	Loss: 0.018868
Train Epoch: 10 [33280/60000 (55%)]	Loss: 0.007889
Train Epoch: 10 [33920/60000 (57%)]	Loss: 0.023195
Train Epoch: 10 [34560/60000 (58%)]	Loss: 0.094016
Train Epoch: 10 [35200/60000 (59%)]	Loss: 0.032923
Train Epoch: 10 [35840/60000 (60%)]	Loss: 0.047655
Train Epoch: 10 [36480/60000 (61%)]	Loss: 0.039834
Train Epoch: 10 [37120/60000 (62%)]	Loss: 0.118570
Train Epoch: 10 [37760/60000 (63%)]	Loss: 0.011477
Train Epoch: 10 [38400/60000 (64%)]	Loss: 0.033962
Train Epoch: 10 [39040/60000 (65%)]	Loss: 0.006781
Train Epoch: 10 [39680/60000 (66%)]	Loss: 0.010087
Train Epoch: 10 [40320/60000 (67%)]	Loss: 0.015013
Train Epoch: 10 [40960/60000 (68%)]	Loss: 0.006516
Train Epoch: 10 [41600/60000 (69%)]	Loss: 0.010866
Train Epoch: 10 [42240/60000 (70%)]	Loss: 0.032060
Train Epoch: 10 [42880/60000 (71%)]	Loss: 0.113746
Train Epoch: 10 [43520/60000 (72%)]	Loss: 0.036509
Train Epoch: 10 [44160/60000 (74%)]	Loss: 0.018629
Train Epoch: 10 [44800/60000 (75%)]	Loss: 0.005781
Train Epoch: 10 [45440/60000 (76%)]	Loss: 0.020277
Train Epoch: 10 [46080/60000 (77%)]	Loss: 0.064707
Train Epoch: 10 [46720/60000 (78%)]	Loss: 0.002237
Train Epoch: 10 [47360/60000 (79%)]	Loss: 0.019705
Train Epoch: 10 [48000/60000 (80%)]	Loss: 0.035027
Train Epoch: 10 [48640/60000 (81%)]	Loss: 0.007895
Train Epoch: 10 [49280/60000 (82%)]	Loss: 0.011368
Train Epoch: 10 [49920/60000 (83%)]	Loss: 0.030328
Train Epoch: 10 [50560/60000 (84%)]	Loss: 0.002525
Train Epoch: 10 [51200/60000 (85%)]	Loss: 0.002397
Train Epoch: 10 [51840/60000 (86%)]	Loss: 0.012999
Train Epoch: 10 [52480/60000 (87%)]	Loss: 0.014599
Train Epoch: 10 [53120/60000 (88%)]	Loss: 0.006732
Train Epoch: 10 [53760/60000 (90%)]	Loss: 0.043986
Train Epoch: 10 [54400/60000 (91%)]	Loss: 0.040023
Train Epoch: 10 [55040/60000 (92%)]	Loss: 0.008028
Train Epoch: 10 [55680/60000 (93%)]	Loss: 0.015426
Train Epoch: 10 [56320/60000 (94%)]	Loss: 0.020987
Train Epoch: 10 [56960/60000 (95%)]	Loss: 0.001457
Train Epoch: 10 [57600/60000 (96%)]	Loss: 0.039079
Train Epoch: 10 [58240/60000 (97%)]	Loss: 0.028353
Train Epoch: 10 [58880/60000 (98%)]	Loss: 0.008176
Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.009452

Test set: Average loss: 0.0410, Accuracy: 9864/10000 (99%)
'''