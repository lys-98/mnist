from __future__ import print_function # 这个是python当中让print都以python3的形式进行print，即把print视为函数
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  #各种优化算法的库
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
'''
Pathlib用于创建路径方式 ，比如有两类如.cwd(当前工作目录)和.home(用户的主目录)：
'''
import time
from model.network.LeNet import LeNet
from model.network.MyNetV1 import MyNetV1
from model.network.MyNetV2 import MyNetV2
from model.network.DefaultNet import DefaultNet
from model.network.MyFullConvNet import MyFullConvNet
from model.network.MyVggNet import MyVggNet
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np

lst =  ["num_1_1.jpg", "num_1_2.jpg",
        "num_2_1.jpg", "num_2_2.jpg",
        "num_3_1.jpg", "num_3_2.jpg",
        "num_4_1.jpg", "num_4_2.jpg",
        "num_5_1.jpg", "num_5_2.jpg",
        "num_6_1.jpg", "num_6_2.jpg",
        "num_7_1.jpg", "num_7_2.jpg",
        "num_8_1.jpg", "num_8_2.jpg",
        "num_9_1.jpg", "num_9_2.jpg"]

def test(model, device):
    model.eval()
    for i in lst:
        img_origin = cv2.imread("./single_test_image/" + i)
        img = cv2.resize(img_origin, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        for i in range(28):
            for j in range(28):
                img[i][j] = 255 - img[i][j]  #?????为什么要用255减去输入图片的像素值
                if img[i][j] < 150:
                    img[i][j] = 0
        print(img)
        exit()
        cv2.imshow("number 3", img_origin)
        #cv2.imshow("number 3", img)
        '''
         cv2.imShow()函数可以在窗口中显示图像。该窗口和图像的原始大小自适应（自动调整到原始尺寸）。

        第一个参数是一个窗口名称（也就是我们对话框的名称），
        它是一个字符串类型。第二个参数是我们的图像。您可以创建任意数量的窗口，但必须使用不同的窗口名称。
        '''

        img = np.array(img).astype(np.float32) #astype：转换数组的数据类型。
        img = np.expand_dims(img, 0) #即扩展维度，np.expand_dims(a,axis=)即在相应的axis轴上扩展维度
        img = np.expand_dims(img, 0) #变成1*1*28*28的图像

        img = torch.from_numpy(img)
        img = img.to(device)
        output = model(Variable(img))
        prob = F.softmax(output, dim=1)
        prob = prob.cpu().detach().numpy()
        #print(prob)
        pred = np.argmax(prob)
        print(pred.item())

        #cv2.imshow("number 3", img)
        cv2.waitKey()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Pytorch MNIST Example")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--model", type=str, default="lenet",
                        help="choose the model to train (default: lenet)")
    '''
    有了 action 这一参数，就相当于把 --no-cuda 参数设成了一个“开关”。我们是不需要给这个开关传递具体的值的，只需要“拨一下”这个开关就行了。

比如上面action='store_true'，就是说当拨动了这个“开关”，就将参数--no-cuda存储为True，
否则参数会存储为False，因为我们设置的 default=False。
    '''
    args = parser.parse_args()

    device = torch.device("cpu")
    
    model_name = args.model.lower()
    if model_name == "lenet":
        model = LeNet().to(device)
    elif model_name == "defaultnet":
        model = DefaultNet().to(device)
    elif model_name == "mynetv1":
        model = MyNetV1().to(device)
    elif model_name == "mynetv2":
        model = MyNetV2().to(device)
    elif model_name == "myfullconvnet":
        model = MyFullConvNet().to(device)
    elif model_name == "myvggnet":
        model = MyVggNet().to(device)
    else:
        print("Wrong model name. Try again!")
        exit()
    
    #model = Net().to(device)
    model_path = Path("./model/weights/{}.pt".format(model_name))
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
    else:
        print("Wrong model name. Try again!")
        exit()
    print("\nTest model:\t{}".format(args.model))
    test(model, device)


if __name__ == "__main__":
    main()
