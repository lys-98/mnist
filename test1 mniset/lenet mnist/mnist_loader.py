from __future__ import print_function
import torch.utils.data as data #读取训练集
from PIL import Image
import os
import os.path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets #自带mnist数据集
from torchvision import transforms
batch_size=test_batch_size=32
class MNIST(data.Dataset):
    # urls = [
    #     'file:///D:\pycharmproject\test1 mniset\lenet mnist\mnist_data\MNIST\mnist_data/train-images-idx3-ubyte.gz',
    #     'file:///D:\pycharmproject\test1 mniset\lenet mnist\mnist_data\MNIST\mnist_data/train-labels-idx1-ubyte.gz',
    #     'file:///D:\pycharmproject\test1 mniset\lenet mnist\mnist_data\MNIST\mnist_data/t10k-images-idx3-ubyte.gz',
    #     # 'file:///E:/PyCharmWorkSpace/Image_Set/mnist_data/t10k-labels-idx1-ubyte.gz',
    #     'file:///D:\pycharmproject\test1 mniset\lenet mnist\mnist_data\MNIST\mnist_data/t10k-labels-idx1-ubyte.gz',
    # ]
    #
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True)

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,  # 测试集
        transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # def __init__(self, root, train=True, transform=None, target_transform=None):
    #     self.root = os.path.expanduser(root) #转化成当前目录
    #     self.transform = transform #可重塑输入数据大小
    #     self.target_transform = target_transform
    #     self.train = train  # training set or test set
    #
    #     if not self._check_exists():
    #         raise RuntimeError('Dataset not found.' +
    #                            ' You can use download=True to download it')
    #
    #     if self.train:
    #         self.train_data, self.train_labels = torch.load(
    #             os.path.join(self.root, self.processed_folder, self.training_file))
    #     else:
    #         self.test_data, self.test_labels = torch.load(
    #             os.path.join(self.root, self.processed_folder, self.test_file))
    #
    # def __getitem__(self, index): #返回目标类的索引
    #     #实例对象的key不管是否存在都会调用类中的__getitem__()方法。
    #     # 而且返回值就是__getitem__()方法中规定的return值。
    #     """
    #     Args:
    #         index (int): Index
    #
    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     if self.train:
    #         img, target = self.train_data[index], self.train_labels[index]
    #     else:
    #         img, target = self.test_data[index], self.test_labels[index]
    #
    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img.numpy(), mode='L')
    #
    #     if self.transform is not None:
    #         img = self.transform(img)
    #
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #
    #     return img, target
    #
    # def __len__(self):
    #     if self.train:
    #         return len(self.train_data)
    #     else:
    #         return len(self.test_data)
    #
    # def _check_exists(self):
    #     return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
    #     os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))
    #

