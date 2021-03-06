import torch.nn as nn
import torch.nn.functional as F
#建立神经网络
class LeNet(nn.Module):
    def __init__(self,channel,classes):
        super(LeNet, self).__init__()
        self.conv1=nn.Conv2d(channel,32,5,1)
        self.conv2=nn.Conv2d(32,64,5,1)
        self.fc1=nn.Linear(4*4*64,512)
        self.fc2=nn.Linear(512,classes)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
