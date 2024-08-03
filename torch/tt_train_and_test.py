
import numpy as np
import torch


def train(model, train_loader, val_loader, val_labels, cuda=False):
    device = "cuda:0" if cuda else "cpu"
    model.to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    train_losses = []
    val_losses = []
    epochs = 15
    val_predictions = []
    for epoch in range(1, epochs + 1):
        val_losses = []
        if epoch % 10 == 0:
            optim.param_groups[0]['lr'] /= 10
        val_predictions = []
        for (images, y_true) in train_loader:
            y_true = y_true.flatten(1).to(device)
            y_pred = model(images.to(device))
            loss = torch.nn.functional.cross_entropy(y_pred, y_true)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.detach().cpu())
        with torch.no_grad():
            for (images, y_true) in val_loader:
                y_true = y_true.flatten(1).to(device)
                y_pred = model(images.to(device))
                loss = torch.nn.functional.cross_entropy(y_pred, y_true)
                val_losses.append(loss.cpu())
                val_predictions.extend(torch.argmax(y_pred, axis=1).cpu())
            print("Val Loss", (sum(val_losses) / len(val_losses)).item())
            print("Accuracy", sum(np.array(val_predictions) == val_labels) / len(val_labels))


def test(model, test_loader, test_labels, cuda=True):
    device = "cuda:0" if cuda else "cpu"
    test_losses = []
    test_predictions = []
    with torch.no_grad():
        for (images, y_true) in test_loader:
            y_true = y_true.flatten(1).to(device)
            y_pred = model(images.to(device))
            loss = torch.nn.functional.cross_entropy(y_pred, y_true)
            test_losses.append(loss.cpu())
            test_predictions.extend(torch.argmax(y_pred, axis=1).cpu())
        print("Accuracy", sum(np.array(test_predictions) == test_labels) / len(test_labels))
        print("Error Rate", sum(np.array(test_predictions) != test_labels) / len(test_labels) * 100)
