from __future__ import print_function # 这个是python当中让print都以python3的形式进行print，即把print视为函数
import argparse
import os
#import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#torch.optim.lr_scheduler模块提供了一些根据epoch训练次数来调整学习率（learning rate）的方法。一般情况下我们会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果。
#而torch.optim.lr_scheduler.ReduceLROnPlateau则提供了基于训练中某些测量值使学习率动态下降的方法。

from pathlib import Path
import time

# import network
from model.network.LeNet import LeNet
from model.network.MyNetV1 import MyNetV1
from model.network.MyNetV2 import MyNetV2
from model.network.DefaultNet import DefaultNet
from model.network.MyFullConvNet import MyFullConvNet
from model.network.MyVggNet import MyVggNet

graph_loss = []
graph_acc = []

def train(args, model, device, train_loader, optimizer, epoch):
    # 这里的train和上面的train不是一个train
    model.train()
    start_time = time.time()
    tmp_time = start_time
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()   # 优化器梯度为什么初始化为0？
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t Cost time: {:.6f}s".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), time.time() - tmp_time
            ))
            tmp_time = time.time()
            graph_loss.append(loss.item())
            if args.dry_run:
                break
    end_time = time.time()
    print("Epoch {} cost {} s".format(epoch, end_time - start_time))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss 
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        float(100. * correct / len(test_loader.dataset))
    ))

    graph_acc.append(100. * correct / len(test_loader.dataset))

# action 和 gamma , metavar的作用
def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Pytorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default : 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default : 1000)")
    parser.add_argument("--epochs", type=int, default=64, metavar="N",
                        help="number of epochs to train (default : 64)")
    parser.add_argument("--learning-rate", type=float, default=0.1, metavar="LR",
                        help="the learning rate (default : 0.1)")
    parser.add_argument("--gamma", type=float, default=0.5, metavar="M",
                        help="Learning rate step gamma (default : 0.5)")
    parser.add_argument("--no-cuda", action="store_true", default=True,
                        help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default : 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--save-model", action = "store_true", default=True,
                        help="For saving the current Model")
    parser.add_argument("--load_state_dict", type=str, default="no",
                        help="load the trained model weights or not (default: no)")
    parser.add_argument("--model", type=str, default="LeNet",
                        help="choose the model to train (default: LeNet)")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available() # not > and > or
    print("user cuda is {}".format(use_cuda))
    torch.manual_seed(args.seed)    # 设置随机种子，什么是随机种子？

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    '''
    *args和**kwargs一般是用在函数定义的时候。二者的意义是允许定义的函数接受任意数目的参数。
    也就是说我们在函数被调用前并不知道也不限制将来函数可以接收的参数数量。
    在这种情况下我们可以使用*args和**kwargs。
    '''
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize(mean, std, inplace=False) mean各通道的均值， std各通道的标准差， inplace是否原地操作
        # 这里说的均值是数据里的均值
        # output = (input - mean) / std
        # 归一化到-1 ~ 1，也不一定，但是属于标准化
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    dataset1 = datasets.MNIST("./data", train=True, download=True,
                            transform=transform)
    dataset2 = datasets.MNIST("./data", train=False,
                            transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

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



    #model = Net().to(device)
    model_path = Path("./model/weights/{}.pt".format(model_name))
    if model_path.exists() and args.load_state_dict == "yes":
        model.load_state_dict(torch.load(model_path))
        print("Load the last trained model.")
    optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
    #optimizer_path = Path("./model/weights/")

    # scheduler是学习率调整，有lambdaLR机制和stepLR机制，lr = lr * gamma^n, n = epoch/step_size
    scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)
    '''
optimizer （Optimizer）：要更改学习率的优化器；
step_size（int）：每训练step_size个epoch，更新一次参数；
gamma（float）：更新lr的乘法因子；
last_epoch （int）：最后一个epoch的index，如果是训练了很多个epoch后中断了，
继续训练，这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1开始。

    '''
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "./model/weights/{}.pt".format(model_name))

    # record the training results
    create_loss_txt_path = "./model/result/{}_loss.txt".format(model_name)
    create_acc_txt_path = "./model/result/{}_acc.txt".format(model_name)
    f = open(create_loss_txt_path, "w+")
    for loss in graph_loss: 
        f.writelines("{}\n".format(loss))
    f.close()
    f = open(create_acc_txt_path, "w+")
    for acc in graph_acc:
        f.writelines("{}\n".format(acc))
    f.close()


if __name__ == "__main__":
    main()

'''
Train Epoch: 64 [0/60000 (0%)]	 Loss: 0.003762	 Cost time: 0.034907s
Train Epoch: 64 [640/60000 (1%)]	 Loss: 0.003183	 Cost time: 0.425862s
Train Epoch: 64 [1280/60000 (2%)]	 Loss: 0.005912	 Cost time: 0.406914s
Train Epoch: 64 [1920/60000 (3%)]	 Loss: 0.030434	 Cost time: 0.433839s
Train Epoch: 64 [2560/60000 (4%)]	 Loss: 0.004577	 Cost time: 0.415888s
Train Epoch: 64 [3200/60000 (5%)]	 Loss: 0.002752	 Cost time: 0.411818s
Train Epoch: 64 [3840/60000 (6%)]	 Loss: 0.001730	 Cost time: 0.541758s
Train Epoch: 64 [4480/60000 (7%)]	 Loss: 0.001244	 Cost time: 0.519876s
Train Epoch: 64 [5120/60000 (9%)]	 Loss: 0.020906	 Cost time: 0.442116s
Train Epoch: 64 [5760/60000 (10%)]	 Loss: 0.035420	 Cost time: 0.453787s
Train Epoch: 64 [6400/60000 (11%)]	 Loss: 0.031730	 Cost time: 0.456778s
Train Epoch: 64 [7040/60000 (12%)]	 Loss: 0.104052	 Cost time: 0.427857s
Train Epoch: 64 [7680/60000 (13%)]	 Loss: 0.035202	 Cost time: 0.404916s
Train Epoch: 64 [8320/60000 (14%)]	 Loss: 0.004275	 Cost time: 0.415888s
Train Epoch: 64 [8960/60000 (15%)]	 Loss: 0.020792	 Cost time: 0.410902s
Train Epoch: 64 [9600/60000 (16%)]	 Loss: 0.050460	 Cost time: 0.433839s
Train Epoch: 64 [10240/60000 (17%)]	 Loss: 0.123646	 Cost time: 0.420876s
Train Epoch: 64 [10880/60000 (18%)]	 Loss: 0.005625	 Cost time: 0.419876s
Train Epoch: 64 [11520/60000 (19%)]	 Loss: 0.005635	 Cost time: 0.423867s
Train Epoch: 64 [12160/60000 (20%)]	 Loss: 0.031361	 Cost time: 0.424865s
Train Epoch: 64 [12800/60000 (21%)]	 Loss: 0.044986	 Cost time: 0.414891s
Train Epoch: 64 [13440/60000 (22%)]	 Loss: 0.007533	 Cost time: 0.497670s
Train Epoch: 64 [14080/60000 (23%)]	 Loss: 0.005380	 Cost time: 0.441818s
Train Epoch: 64 [14720/60000 (25%)]	 Loss: 0.021638	 Cost time: 0.420876s
Train Epoch: 64 [15360/60000 (26%)]	 Loss: 0.016277	 Cost time: 0.404917s
Train Epoch: 64 [16000/60000 (27%)]	 Loss: 0.014922	 Cost time: 0.518614s
Train Epoch: 64 [16640/60000 (28%)]	 Loss: 0.023865	 Cost time: 0.430847s
Train Epoch: 64 [17280/60000 (29%)]	 Loss: 0.003061	 Cost time: 0.412896s
Train Epoch: 64 [17920/60000 (30%)]	 Loss: 0.008889	 Cost time: 0.415888s
Train Epoch: 64 [18560/60000 (31%)]	 Loss: 0.011328	 Cost time: 0.423865s
Train Epoch: 64 [19200/60000 (32%)]	 Loss: 0.046521	 Cost time: 0.424865s
Train Epoch: 64 [19840/60000 (33%)]	 Loss: 0.015237	 Cost time: 0.429851s
Train Epoch: 64 [20480/60000 (34%)]	 Loss: 0.001273	 Cost time: 0.452791s
Train Epoch: 64 [21120/60000 (35%)]	 Loss: 0.004157	 Cost time: 0.487695s
Train Epoch: 64 [21760/60000 (36%)]	 Loss: 0.002447	 Cost time: 0.505642s
Train Epoch: 64 [22400/60000 (37%)]	 Loss: 0.031085	 Cost time: 0.441927s
Train Epoch: 64 [23040/60000 (38%)]	 Loss: 0.005687	 Cost time: 0.554518s
Train Epoch: 64 [23680/60000 (39%)]	 Loss: 0.045542	 Cost time: 0.461765s
Train Epoch: 64 [24320/60000 (41%)]	 Loss: 0.002834	 Cost time: 0.409905s
Train Epoch: 64 [24960/60000 (42%)]	 Loss: 0.001196	 Cost time: 0.413894s
Train Epoch: 64 [25600/60000 (43%)]	 Loss: 0.002026	 Cost time: 0.424863s
Train Epoch: 64 [26240/60000 (44%)]	 Loss: 0.007023	 Cost time: 0.420875s
Train Epoch: 64 [26880/60000 (45%)]	 Loss: 0.073339	 Cost time: 0.425862s
Train Epoch: 64 [27520/60000 (46%)]	 Loss: 0.065955	 Cost time: 0.410901s
Train Epoch: 64 [28160/60000 (47%)]	 Loss: 0.014954	 Cost time: 0.419877s
Train Epoch: 64 [28800/60000 (48%)]	 Loss: 0.005673	 Cost time: 0.415888s
Train Epoch: 64 [29440/60000 (49%)]	 Loss: 0.041159	 Cost time: 0.413893s
Train Epoch: 64 [30080/60000 (50%)]	 Loss: 0.010826	 Cost time: 0.477723s
Train Epoch: 64 [30720/60000 (51%)]	 Loss: 0.045586	 Cost time: 0.439825s
Train Epoch: 64 [31360/60000 (52%)]	 Loss: 0.018106	 Cost time: 0.415888s
Train Epoch: 64 [32000/60000 (53%)]	 Loss: 0.004761	 Cost time: 0.402504s
Train Epoch: 64 [32640/60000 (54%)]	 Loss: 0.003250	 Cost time: 0.501374s
Train Epoch: 64 [33280/60000 (55%)]	 Loss: 0.014541	 Cost time: 0.455781s
Train Epoch: 64 [33920/60000 (57%)]	 Loss: 0.003764	 Cost time: 0.466752s
Train Epoch: 64 [34560/60000 (58%)]	 Loss: 0.003593	 Cost time: 0.432842s
Train Epoch: 64 [35200/60000 (59%)]	 Loss: 0.051986	 Cost time: 0.409904s
Train Epoch: 64 [35840/60000 (60%)]	 Loss: 0.049130	 Cost time: 0.434841s
Train Epoch: 64 [36480/60000 (61%)]	 Loss: 0.001296	 Cost time: 0.411896s
Train Epoch: 64 [37120/60000 (62%)]	 Loss: 0.021572	 Cost time: 0.417882s
Train Epoch: 64 [37760/60000 (63%)]	 Loss: 0.020021	 Cost time: 0.415889s
Train Epoch: 64 [38400/60000 (64%)]	 Loss: 0.016815	 Cost time: 0.404917s
Train Epoch: 64 [39040/60000 (65%)]	 Loss: 0.000857	 Cost time: 0.429850s
Train Epoch: 64 [39680/60000 (66%)]	 Loss: 0.037908	 Cost time: 0.419878s
Train Epoch: 64 [40320/60000 (67%)]	 Loss: 0.008418	 Cost time: 0.405916s
Train Epoch: 64 [40960/60000 (68%)]	 Loss: 0.039110	 Cost time: 0.404917s
Train Epoch: 64 [41600/60000 (69%)]	 Loss: 0.015651	 Cost time: 0.417883s
Train Epoch: 64 [42240/60000 (70%)]	 Loss: 0.001293	 Cost time: 0.412896s
Train Epoch: 64 [42880/60000 (71%)]	 Loss: 0.012855	 Cost time: 0.429850s
Train Epoch: 64 [43520/60000 (72%)]	 Loss: 0.097092	 Cost time: 0.431846s
Train Epoch: 64 [44160/60000 (74%)]	 Loss: 0.003029	 Cost time: 0.419877s
Train Epoch: 64 [44800/60000 (75%)]	 Loss: 0.028453	 Cost time: 0.428854s
Train Epoch: 64 [45440/60000 (76%)]	 Loss: 0.051722	 Cost time: 0.408906s
Train Epoch: 64 [46080/60000 (77%)]	 Loss: 0.016251	 Cost time: 0.429851s
Train Epoch: 64 [46720/60000 (78%)]	 Loss: 0.121582	 Cost time: 0.400928s
Train Epoch: 64 [47360/60000 (79%)]	 Loss: 0.034656	 Cost time: 0.415889s
Train Epoch: 64 [48000/60000 (80%)]	 Loss: 0.012168	 Cost time: 0.413893s
Train Epoch: 64 [48640/60000 (81%)]	 Loss: 0.010917	 Cost time: 0.424864s
Train Epoch: 64 [49280/60000 (82%)]	 Loss: 0.005081	 Cost time: 0.432843s
Train Epoch: 64 [49920/60000 (83%)]	 Loss: 0.045924	 Cost time: 0.433839s
Train Epoch: 64 [50560/60000 (84%)]	 Loss: 0.009340	 Cost time: 0.423867s
Train Epoch: 64 [51200/60000 (85%)]	 Loss: 0.105661	 Cost time: 0.422869s
Train Epoch: 64 [51840/60000 (86%)]	 Loss: 0.015073	 Cost time: 0.418880s
Train Epoch: 64 [52480/60000 (87%)]	 Loss: 0.000980	 Cost time: 0.416885s
Train Epoch: 64 [53120/60000 (88%)]	 Loss: 0.007706	 Cost time: 0.419878s
Train Epoch: 64 [53760/60000 (90%)]	 Loss: 0.045229	 Cost time: 0.451792s
Train Epoch: 64 [54400/60000 (91%)]	 Loss: 0.011525	 Cost time: 0.436833s
Train Epoch: 64 [55040/60000 (92%)]	 Loss: 0.003330	 Cost time: 0.419877s
Train Epoch: 64 [55680/60000 (93%)]	 Loss: 0.032481	 Cost time: 0.449609s
Train Epoch: 64 [56320/60000 (94%)]	 Loss: 0.022661	 Cost time: 0.437400s
Train Epoch: 64 [56960/60000 (95%)]	 Loss: 0.002876	 Cost time: 0.496507s
Train Epoch: 64 [57600/60000 (96%)]	 Loss: 0.052141	 Cost time: 0.471740s
Train Epoch: 64 [58240/60000 (97%)]	 Loss: 0.002835	 Cost time: 0.555991s
Train Epoch: 64 [58880/60000 (98%)]	 Loss: 0.001160	 Cost time: 0.564333s
Train Epoch: 64 [59520/60000 (99%)]	 Loss: 0.001172	 Cost time: 0.457351s
Epoch 64 cost 41.100255489349365 s

Test set: Average loss: 0.0359, Accuracy: 9891/10000 (98.91%)
'''