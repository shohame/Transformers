import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_dataset_loader(batch_size=32, val_split=0.1):
    # Define the transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)

    # Split the training data into train and validation sets
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Download and load the test data
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_dataset_as_tensors():
    # Load the entire dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    # Convert to tensors
    train_images = torch.stack([img for img, _ in train_dataset])
    train_labels = torch.tensor([label for _, label in train_dataset])
    test_images = torch.stack([img for img, _ in test_dataset])
    test_labels = torch.tensor([label for _, label in test_dataset])

    # Split train into train and validation
    val_size = int(len(train_images) * 0.1)
    train_size = len(train_images) - val_size

    train_images, val_images = torch.split(train_images, [train_size, val_size])
    train_labels, val_labels = torch.split(train_labels, [train_size, val_size])

    # Convert labels to one-hot encoding
    def to_onehot(labels, num_classes=10):
        return torch.eye(num_classes)[labels]

    train_labels_onehot = to_onehot(train_labels)
    val_labels_onehot = to_onehot(val_labels)
    test_labels_onehot = to_onehot(test_labels)

    return (train_images, train_labels_onehot,
            val_images, val_labels_onehot,
            test_images, test_labels_onehot,
            val_labels, test_labels)


# Example usage
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataset_loader()
    print(
        f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # If you need the dataset as tensors
    (train_images, train_labels, val_images, val_labels,
     test_images, test_labels, val_labels_raw, test_labels_raw) = get_dataset_as_tensors()

    print(f"Train images shape: {train_images.shape}, Train labels shape: {train_labels.shape}")
    print(f"Validation images shape: {val_images.shape}, Validation labels shape: {val_labels.shape}")
    print(f"Test images shape: {test_images.shape}, Test labels shape: {test_labels.shape}")