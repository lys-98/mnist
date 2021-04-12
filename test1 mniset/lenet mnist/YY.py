import mnist_loader
import torch
import numpy as np
test_loader = mnist_loader.MNIST.test_loader
for  data, target in test_loader:
    data, target = data, target
    data_record=data[0:3]
    target_record = target[0:10].cpu().numpy()
    print(data_record)
