
import matplotlib.pyplot as plt
import torch
from tt_dataset import get_dataset_loader
from tt_ViT_model import ViT
import random
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


    vit = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
              dim=64, depth=6, heads=8, mlp_dim=128).cuda()

    print("Parameters:", count_parameters(vit))
    train(vit, train_loader, val_loader, val_labels, cuda=True)
    print("-" * 40)
    print('Test:')
    test(vit, test_loader, test_labels, cuda=True)


    fig = plt.Figure((8, 8))
    for index in range(1, 10):
        plt.subplot(3,3,index)
        i = random.randint(0, len(torch_val_images))
        plt.imshow(torch_val_images[i], cmap="bone")
        plt.axis("off")
        plt.title(f"Prediction: {torch.argmax(vit(torch_val_images[i][None,None,:,:].cuda()).cpu(), axis=1).item()}")

    plt.show()



if __name__ == "__main__":
    main()

