import torch
from torch import nn,optim  #optim包含一些更新参数的优化算法，如SGD，ADAGrad，adam等
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader #DataLoader构建可迭代的数据装载器
from torchvision import transforms #transforms：提供常用的数据预处理操作
from torchvision import datasets #datasets： 提供常用的数据集加载，设计上都是继承
# torch.utils.data.Dataset，主要包括 MNIST、ImageNet、COCO等；
import os #在python环境下对文件，文件夹执行操作的一个模块。
import numpy as np
from PIL import Image #图像处理的库

batch_size = 200    # 分批训练数据、每批数据量
learning_rate = 1e-2    # 学习率
num_epoches = 2    # 训练次数
DOWNLOAD_MNIST = True    # 是否网上下载数据

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    ##其中os.listdir()用于返回一个由文件名和目录名组成的列
    # 判断该目录下是否包含该文件
    DOWNLOAD_MNIST = True

train_dataset = datasets.MNIST(
    root = './mnist',
    train= True,
    transform = transforms.ToTensor(), #将输入的图片转化成tensor
    download=DOWNLOAD_MNIST
)
test_dataset = datasets.MNIST(
    root='./mnist',
    train=False,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#DataLoader构建可迭代的数据装载器； shuffle：每个epoch是否乱序；

class CNN(nn.Module):
    def __init__(self,in_dim,n_class):
        super(CNN,self).__init__()
        # nn.Sequential() 这个表示将一个有序的模块写在一起，
        # 也就相当于将神经网络的层按顺序放在一起，这样可以方便结构显示
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim,6,kernel_size=3,stride=1,padding=1),
            # input shape(1*28*28),(28+1*2-3)/1+1=28 卷积后输出（6*28*28）
            # 输出图像大小计算公式:(n*n像素的图）(n+2p-k)/s+1
            nn.ReLU(True),        # 激活函数
            nn.MaxPool2d(2,2),    # 28/2=14 池化后（6*14*14）
            nn.Conv2d(6,16,5,stride=1,padding=0),  # (14-5)/1+1=10 卷积后（16*10*10）
            nn.ReLU(True),
            nn.MaxPool2d(2,2)    #池化后（16*5*5）=400，the input of full connection
        )
        self.fc = nn.Sequential(   #full connection layers.
            nn.Linear(400,120),
            nn.Linear(120,84),
            nn.Linear(84,n_class)
        )

    def forward(self, x):
        out = self.conv(x)                  #out shape(batch,16,5,5)
        out = out.view(out.size(0), -1)     #out shape(batch,400)
        out = self.fc(out)                  #out shape(batch,10)
        return out


cnn = CNN(1, 10)
print(cnn)

if torch.cuda.is_available():       #是否可用GPU计算
     cnn = cnn.cuda()           #转换成可用GPU计算的模型

criterion = nn.CrossEntropyLoss()       #多分类用的交叉熵损失函数
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
#常用优化方法有
#1.Stochastic Gradient Descent (SGD)
#2.Momentum
#3.AdaGrad
#4.RMSProp
#5.Adam (momentum+adaGrad)   效果较好

for epoch in range(num_epoches):
    print('epoch{}'.format(epoch+1))
    print('*'*10)
    running_loss = 0.0
    running_acc = 0.0
    #训练
    for i,data in enumerate(train_loader,1):  #enumerate获得值和索引的一个容器
        img,label = data
        #  判断是否可以使用GPU，若可以则将数据转化为GPU可以处理的格式。
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = cnn(img)
        loss = criterion(out,label)
        # running_loss += loss.item()
        running_loss += loss.item() * label.size(0) #计算所有batch的损失和
        #label.size(0)返回多少个lable
        _, pred = torch.max(out,1)
        # torch.max()这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示），
        # 第二个值是value所在的index（也就是predicted）。
        num_correct = (pred == label).sum()
        # accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Finish {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(
        epoch+1,running_loss/(len(train_dataset)),running_acc/len(train_dataset)
    ))

#测试
cnn.eval()
eval_loss = 0
eval_acc = 0
for i, data in enumerate(test_loader, 1):
    img, label = data
    #判断是否可以使用GPU，若可以则将数据转化为GPU可以处理的格式。
    if torch.cuda.is_available():
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        img = Variable(img)
        label = Variable(label)

    out = cnn(img)
    loss = criterion(out,label)
    eval_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    accuracy = (pred == label).float().mean()
    eval_acc += num_correct.item()

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc/len(test_dataset)))


# Save the Trained Model
# ckpt_dir = 'D:/'
ckpt_dir = 'D:\pycharmproject\mniset cnn\mnist_data'
save_path = os.path.join(ckpt_dir, 'CNN_model_weight2.pth.tar')
torch.save({'state_dict': cnn.state_dict()}, save_path) #保存绝对路径
#state_dict 将cnn的每一层的W，b保存下来


#  加载参数
ckpt = torch.load('D:\pycharmproject\mniset cnn\mnist_data\CNN_model_weight2.pth.tar')
cnn.load_state_dict(ckpt['state_dict'])            #参数加载到指定模型cnn
#  要识别的图片
input_image = 'D:\pycharmproject\mniset cnn\mnist_data/6.png'

im = Image.open(input_image).resize((28, 28))     #取图片数据
im = im.convert('L')      #灰度图
im_data = np.array(im) #将输入图片转化为矩阵
im_data = torch.from_numpy(im_data).float()
im_data = im_data.view(1, 1, 28, 28)
out = cnn(im_data)
_, pred = torch.max(out, 1)

print('预测为:数字{}。'.format(int(pred)))
'''
注：Pytorch基于nn.Module构建的模型中，只支持mini-batch的Variable输入方式，
    比如，只有一张输入图片，也需要变成 N x C x H x W 的形式：    
    input_image = torch.FloatTensor(1, 28, 28)   
    input_image = input_image.unsqueeze(0)   # 1 x 1 x 28 x 28
'''

'''
CNN(
  (conv): Sequential(
    (0): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Linear(in_features=400, out_features=120, bias=True)
    (1): Linear(in_features=120, out_features=84, bias=True)
    (2): Linear(in_features=84, out_features=10, bias=True)
  )
)
epoch1
**********
Finish 1 epoch,Loss:0.233514,Acc:0.925567
epoch2
**********
Finish 2 epoch,Loss:0.084307,Acc:0.973417
Test Loss: 0.068168, Acc: 0.978700
预测为:数字6。

Process finished with exit code 0


'''