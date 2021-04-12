import matplotlib
import os
import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys
'''
argparse 是 Python 内置的一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，
argparse库是一个存储参数库，可以用来进行模型训练过程中的参数保存作为一个整体，以便于使用和更改。
 主要有三个步骤：
创建 ArgumentParser() 对象
调用 add_argument() 方法添加参数
使用 parse_args() 解析添加的参数
'''
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="lenet")
args = parser.parse_args()

#file_loss_path = "E:/WorkSpace/Pytorch/mnist/model/result/{}_loss.txt".format(args.model)
file_loss_path = sys.path[0] + "/model/result/{}_loss.txt".format(args.model)
#python从sys.path列表的第一个目录开始按顺序检查目录，直到找到它要查找的.py文件。
#sys.path是python的搜索模块的路径集，返回的结果是一个list
#path[0]是空字符串，它引导Python首先在当前目录中搜索模块。
lst_loss = list()
with open(file_loss_path) as file_object:
    for line in file_object: #以每一行的方式检查文件
        if "e" in line:
            lst_loss.append(eval(line))
            #eval() 函数作用：可以接受一个字符串str作为参数，并把这个参数作为脚本代码来执行。
        else:
            lst_loss.append(float(line[:-2]))
    file_object.close()
# print(lst_loss)

#file_acc_path = "E:/WorkSpace/Pytorch/mnist/model/result/{}_acc.txt".format(args.model)
file_acc_path = sys.path[0] +  "/model/result/{}_acc.txt".format(args.model)
lst_acc = list()
with open(file_acc_path) as file_object:
    for line in file_object:
        if "e" in line:
            lst_acc.append(eval(line))
        else:
            lst_acc.append(float(line[:-2]))
    file_object.close()
print(lst_acc)

plt.title("{} loss".format(args.model))
plt.plot(lst_loss)
plt.xlim(0 - len(lst_loss) / 20, len(lst_loss))
plt.ylim(0, 1.5)
plt.grid() #绘制网格线
plt.savefig(file_loss_path[:-3] + "jpg") #保存固定路径

plt.title("{} acc".format(args.model))
plt.plot(lst_acc)
plt.xlim(0 - len(lst_acc) / 20, len(lst_acc))
plt.ylim(min(lst_acc) - 1, max(max(lst_acc) + 1, 100))
plt.savefig(file_acc_path[:-3] + "jpg")