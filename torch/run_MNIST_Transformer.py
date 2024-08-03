
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tt_dataset import get_dataset_loader
from tt_MNIST_Transformer_model import MNISTTransformer

from tt_train_and_test import train, test

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    (train_loader,
     val_loader,
     test_loader,
     val_labels,
     test_labels,
     torch_val_images)  = get_dataset_loader()

    print (f'Running: model = MNISTTransformer(1)')
    print('--------------------------------------')
    model = MNISTTransformer(1)
    print("Parameters:", count_parameters(model))
    train(model, train_loader, val_loader, val_labels, cuda=True)
    print("-" * 40)
    print('Test:')
    test(model, test_loader, test_labels, cuda=True)

    print (f'/n ===================================================================== /n')
    print (f'Running: model = MNISTTransformer(10)')
    print('-' * 40)

    model = MNISTTransformer(10)
    print("Parameters:", count_parameters(model))
    train(model, train_loader, val_loader, val_labels, cuda=True)
    print("-" * 40)
    print('Test:')
    test(model, test_loader, test_labels, cuda=True)

    print (f'/n ===================================================================== /n')
    print (f'Running: model = MNISTTransformer(20)')
    print('_' * 20)

    model = MNISTTransformer(20)
    print("Parameters:", count_parameters(model))
    train(model, train_loader, val_loader, val_labels, cuda=True)
    print("-" * 40)
    print('Test:')
    test(model, test_loader, test_labels, cuda=True)




if __name__ == "__main__":
    main()

