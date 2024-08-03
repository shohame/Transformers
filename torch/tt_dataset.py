
from pathlib import Path
from array import array
import struct
import numpy as np
import matplotlib.pyplot as plt
import torch



class Dataset(torch.utils.data.Dataset):

    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels
        self.add_channel_f = lambda x: x[None,:,:]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.add_channel_f(self.images[idx]), self.labels[idx]

def get_dataset_loader():
    (torch_train_images,
     torch_train_labels,
     torch_val_images,
     torch_val_labels,
     torch_test_images,
     torch_test_labels,
     val_labels,
     test_labels
     ) = get_dataset_as_tensors()

    train_dataset = Dataset(torch_train_images, torch_train_labels)
    val_dataset = Dataset(torch_val_images, torch_val_labels)
    test_dataset = Dataset(torch_test_images, torch_test_labels)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    return train_loader, val_loader, test_loader, val_labels, test_labels, torch_val_images
def read_image_data(path):
    data_dir = Path("../mnist-dataset/")
    with open(data_dir / path, "rb") as f:
        # IDX file format
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = array("B", f.read())
    images = []
    for i in range(size):
        image = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28, 28)
        images.append(image)
    return np.array(images)


def read_labels(path):
    data_dir = Path('../mnist-dataset/')
    with open(data_dir / path, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = np.array(array("B", f.read()))
    return labels


def get_dataset_as_tensors():
    train_val_labels = read_labels("train-labels-idx1-ubyte")
    assert train_val_labels[0] == 5

    train_val_images = read_image_data("train-images-idx3-ubyte")
    test_images = read_image_data("t10k-images-idx3-ubyte")

    train_images = train_val_images[:int(0.9 * len(train_val_images))]
    val_images = train_val_images[len(train_images):]
    assert len(train_images) + len(val_images) == len(train_val_images)

    train_labels = train_val_labels[:len(train_images)]
    val_labels = train_val_labels[len(train_images):]
    test_labels = read_labels("t10k-labels-idx1-ubyte")
    assert len(train_labels) + len(val_labels) == len(train_val_labels)
    assert test_labels[0] == 7

    def onehot(label):
        encoded_label = np.zeros(10)
        encoded_label[label] = 1
        return encoded_label

    torch_train_images = torch.Tensor(train_images)
    torch_val_images = torch.Tensor(val_images)
    torch_test_images = torch.Tensor(test_images)

    torch_train_labels = torch.Tensor([onehot(label) for label in train_labels])
    torch_val_labels = torch.Tensor([onehot(label) for label in val_labels])
    torch_test_labels = torch.Tensor([onehot(label) for label in test_labels])

    return (torch_train_images,
            torch_train_labels,
            torch_val_images,
            torch_val_labels,
            torch_test_images,
            torch_test_labels, val_labels, test_labels)
