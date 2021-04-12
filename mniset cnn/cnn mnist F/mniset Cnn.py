import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets #提供常用的数据不加载，主要包含mnist、amagenet、coco等
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#预置参数
torch.manual_seed(1)
batch_size=128
learning_rate=1e-2
num_epoches=1

#加载数据
train_dataset=datasets.MNIST(root="./mnist_data",
                             train=True,  #如果为True，则从training.pt创建数据集，否则从test.pt创建数据集
                             transform=transforms.ToTensor(),#将输入的图片转化成tensor
                             download=True)
test_dataset=datasets.MNIST(root="./mnist_data",
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
#DataLoader构建可迭代的数据装载器； shuffle：每个epoch是否乱序；

#创建模型
class Cnn(nn.Module):
    def __init__(self,in_dim,n_class):
        super(Cnn,self).__init__()
        #nn.Sequential() 这个表示将一个有序的模块写在一起，也就相当于将神经网络的层按顺序放在一起，这样可以方便结构显示
        self.conv=nn.Sequential(nn.Conv2d(in_dim,6,3,stride=1,padding=1),#(28-3+2*1)/1+1=28                                nn.ReLU(True),
                                nn.MaxPool2d(2,2),#14*14
                                nn.Conv2d(6,16,5,stride=1,padding=0),#(14-5+2*0)/1+1=10
                                nn.ReLU(True),
                                nn.MaxPool2d(2,2))  #5*5*16=400

        self.fc=nn.Sequential(nn.Linear(400,120),
                               nn.Linear(120,84),
                               nn.Linear(84,n_class))

    def forward(self,x):
        out=self.conv(x)
        out=out.view(out.size(0),400)
        out=self.fc(out)
        return out

model=Cnn(1,10)
print(model)
#
# #训练模型
criterion=nn.CrossEntropyLoss() #交叉熵 计算loss的一种方法
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate) #SGD是梯度下降的优化器，加入了动量

print("train: ")
for epoch in range(num_epoches):
    running_loss=0.0
    running_acc=0.0
    for i,data in enumerate(train_loader,0):
        img,label=data
        out=model(img)
        loss=criterion(out,label)
        # running_loss+=loss.item()*label.size(0) #???
        running_loss += loss.item()
        _,pred=torch.max(out,1) #orch.max()这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示），
        # 第二个值是value所在的index（也就是predicted）。
        num_correct=(pred==label).sum()
        running_acc+=num_correct.item()

        optimizer.zero_grad() #将模型的所有参数的梯度清零.
        loss.backward()
        optimizer.step()
    print(epoch+1," loss:",running_loss/len(train_dataset)," acc:",running_acc/len(train_dataset))



#模型测试
# model.eval()
# eval_loss=0
# eval_acc=0
# for data in test_loader:
#
#     out=model(img)
#     loss=criterion(out,label)
#     eval_loss+=loss.item()*label.size(0)
#     _,pred=torch.max(out,1)
#     num_correct=(pred==label).sum()
#     eval_acc+=num_correct.item()


# def test(model, device, test_loader):
def test(model, test_loader):
    model.eval()  # 进入测试模式
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            data, target = data, target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # test_loss += loss.item() * label.size(0)

            pred = output.argmax(dim=1, keepdim=True)
            #keepdim 表示是否需要保持输出的维度与输入一样，keepdim=True表示输出和输入的维度一样，
            # keepdim=False表示输出的维度被压缩了，也就是输出会比输入低一个维度。
            # _, pred = torch.max(out, 1)
            data_record=data[0:10]
            pred_record=pred.view_as(target)[0:10].cpu().numpy()
            #返回target的size；
            target_record=target[0:10].cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    return data_record,pred_record,target_record
data_record,pred_record,target_record=test(model, test_loader)
#可视化分类结果
label_dict={0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9"}
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig=plt.gcf()
    fig.set_size_inches(12,6)
    if num>10:
        num=10
    for i in range(0,num):
        image = images[idx].cpu().clone()
        image = image.squeeze(0)
        #image = unloader(image)
        ax=plt.subplot(2,5,1+i)
        ax.imshow(image,cmap="binary") #图像展示为灰白色
        title=label_dict[labels[idx]]
        if len(prediction)>0:
            title+="=>"+label_dict[prediction[idx]]
        ax.set_title(title,fontsize=10)
        idx+=1
    plt.show()
plot_images_labels_prediction(data_record,target_record,pred_record,0,10)



