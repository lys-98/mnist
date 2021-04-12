import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import mnist_loader
import Build_LeNet_for_mnist
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import csv
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets
#加载数据集
# use_cuda=torch.cuda.is_available()##检测显卡是否可用
batch_size=test_batch_size=32
# kwargs={'num_workers':0,'pin_memory':True}if use_cuda else {}
# kwargs={'num_workers':0,'pin_memory':True}
#训练数据加载
# train_loader = torch.utils.data.DataLoader(
#     mnist_loader.MNIST('./mnist_data',
#                    train=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),  # 第一个参数dataset：数据集
#     batch_size=batch_size,
#     shuffle=True,  # 随机打乱数据
#     )  #   原来是**kwargs)#kwargs是上面gpu的设置
# #测试数据加载
# test_loader = torch.utils.data.DataLoader(
#     mnist_loader.MNIST('./mnist_data',
#                    train=False,  # 如果False，从test.pt创建数据集
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=test_batch_size,
#     shuffle=True,
#     ) # **kwargs)
#修改后1
# train_dataset=datasets.MNIST(root="./mnist_data",
#                              train=True,
#                              transform=transforms.ToTensor(),
#                              download=True)
# test_dataset=datasets.MNIST(root="./mnist_data",
#                             train=False,
#                             transform=transforms.ToTensor(),
#                             download=True)
# train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
# test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

#修改后2
train_loader = mnist_loader.MNIST.train_loader
test_loader = mnist_loader.MNIST.test_loader
#加入神经网络及参数设置
learning_rate=0.01
momentum=0.9
# device = torch.device("cuda" if use_cuda else "cpu")
# model=Build_LeNet_for_mnist.LeNet(1, 10).to(device)#加载模型
model=Build_LeNet_for_mnist.LeNet(1, 10)
optimizer=optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)#优化器选择

#创建csv文件
csvFile = open("log.csv", "a+")
writer = csv.writer(csvFile)    #创建写的对象
last_epoch=0
if os.path.exists("cifar10_cnn.pt"):
    print("load pretrain")
    model.load_state_dict(torch.load("cifar10_cnn.pt"))
    data = pd.read_csv('log.csv')
    e = data['epoch']
    last_epoch=e[len(e)-1]
else:
    print("first train")
    #先写入columns_name
    writer.writerow(["epoch","acc","loss"])

#训练函数
def train(model, train_loader, optimizer, last_epoch,epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    print("Train from Epoch: {}".format(last_epoch+1))
    model.train()  # 进入训练模式
    for epoch in range(1+last_epoch, epochs + 1+last_epoch):
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # data, target = data.to(device), target.to(device)
            data, target = data, target
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc=100. * correct / len(train_loader.dataset)
        print("Train Epoch: {} Accuracy:{:0f}% Loss: {:.6f}".format(
            epoch,
            acc,
            loss.item()
        ))
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            #print(model.state_dict())
        writer.writerow([epoch,acc/100,loss.item()])
    return(best_model_wts)
#开始训练和测试
epochs = 1
best_model_wts=train(model,  train_loader, optimizer,last_epoch, epochs)

csvFile.close()
#保存训练模型
save_model = True
if (save_model):
    torch.save(best_model_wts,"mnist_LeNet.pt")
    #词典格式，model.state_dict()只保存模型参数

#可视化准确率
data = pd.read_csv('log.csv')
epoch = data['epoch']
acc = data['acc']
loss = data['loss']

fig=plt.gcf()
fig.set_size_inches(10,4)
plt.title("Accuracy&Loss")
plt.xlabel("Training Epochs")
plt.ylabel("Value")
plt.plot(epoch,acc,label="Accuracy")
#plt.plot(epoch,loss,label="Loss")
plt.ylim((0,1.))
# plt.xticks(np.arange(1, len(epoch+1), 1.0))
# plt.xticks(np.arange(1.0, len(epoch+1.0), 1.0))
# plt.yticks(np.arange(0.0, 1.5, 0.2))
plt.legend()
plt.show()


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
            pred = output.argmax(dim=1, keepdim=True)
            data_record=data[0:10]
            pred_record=pred.view_as(target)[0:10].cpu().numpy()
            target_record=target[0:10].cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    return data_record,pred_record,target_record
data_record,pred_record,target_record=test(model, test_loader)

#可视化测试分类结果
#unloader = transforms.ToPILImage()
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
        ax.imshow(image,cmap="binary")
        title=label_dict[labels[idx]]
        if len(prediction)>0:
            title+="=>"+label_dict[prediction[idx]]
        ax.set_title(title,fontsize=10)
        idx+=1
    plt.show()
plot_images_labels_prediction(data_record,target_record,pred_record,0,10)

'''
训练结果
first train
Train from Epoch: 1
Train Epoch: 1 Accuracy:93.751667%	Loss: 0.025602
Train Epoch: 2 Accuracy:98.533333%	Loss: 0.018964
Train Epoch: 3 Accuracy:99.008333%	Loss: 0.011548
Train Epoch: 4 Accuracy:99.275000%	Loss: 0.044463
Train Epoch: 5 Accuracy:99.400000%	Loss: 0.015904
Train Epoch: 6 Accuracy:99.561667%	Loss: 0.001069
Train Epoch: 7 Accuracy:99.701667%	Loss: 0.000348
Train Epoch: 8 Accuracy:99.791667%	Loss: 0.009727
Train Epoch: 9 Accuracy:99.770000%	Loss: 0.001520
Train Epoch: 10 Accuracy:99.836667%	Loss: 0.000392
'''