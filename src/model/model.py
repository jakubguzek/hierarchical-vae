#!/usr/bin/env python
import pathlib
import sys
from typing import Any, Type

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torchvision import transforms, datasets

import process_data

HYPERPARAMETERS = {"batch_size": 128, "learning_rate": 1.0e-3, "use_cuda": False, "epochs": 10}


class Encoder(nn.Module):
    def __init__(self, hidden_size, latent_size) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 48, 11, stride=4)
        self.conv2 = nn.Conv2d(48, 24, 7, stride=2)
        self.flatten = nn.Flatten(start_dim=0)
        self.a1 = nn.ReLU()
        self.fc1 = nn.Linear(13824, hidden_size)
        self.a2 = nn.ReLU()
        self.z_mean = nn.Linear(hidden_size, latent_size)
        self.z_logvar = nn.Linear(hidden_size, latent_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.a1(self.flatten(x))
        x = self.a2(self.fc1(x))
        z_mean = self.sigmoid(self.z_mean(x))
        z_logvar = self.z_logvar(x)
        return z_mean, z_logvar


class Decoder(nn.Module):
    def __init__(self, hidden_size, latent_size) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 13824)
        self.a1 = nn.ReLU()
        self.unflatten = nn.Unflatten(1, (24, 24, 24))
        self.a2 = nn.ReLU()
        self.convt1 = nn.ConvTranspose2d(24, 48, 7, stride=2)
        self.convt2 = nn.ConvTranspose2d(48, 3, 11, stride=4)
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, z):
        x = self.a1(self.fc1(z))
        x = self.a2(self.fc2(x))
        x = self.unflatten(x)
        x = self.convt2(self.convt1(x))
        x = self.logsigmoid(x)
        return x


class bVAE(nn.Module):
    def __init__(self, hidden_size, latent_size, beta=1) -> None:
        super().__init__()
        self.z_dim = latent_size
        self.encoder = Encoder(hidden_size, latent_size)
        self.decoder = Decoder(hidden_size, latent_size)
        self.beta = beta

    def elbo_loss(self, x, z_mean, z_logvar, x_prime):
        reconstruction_loss = x_prime.sum(dim=(1, 2, 3)).mean()
        kld = 0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        kld_loss = kld.mean()
        return self.beta * -reconstruction_loss + kld_loss

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)

        # Sample with reparametrization trick
        z = distributions.Normal(z_mean, torch.exp(z_logvar / 2)).rsample((1,))  # type: ignore
        # print(z.shape)
        # print(z)
        # z = z.reshape((z.shape[1], self.z_dim))

        x_prime = self.decoder(z)
        return z_mean, z_logvar, x_prime

    def sample(self):
        z = distributions.Normal(torch.zeros(self.z_dim)).sample((1,))  # type; ignore
        return self.decoder(z)


def train_one_epoch(
    model: nn.Module, optimizer_type: Type[torch.optim.Optimizer], train_loader
):
    running_loss = 0.0
    last_loss = 0.0

    optimizer = optimizer_type(
        list(model.parameters()), lr=HYPERPARAMETERS["learning_rate"]
    )

    for i, data in enumerate(train_loader):
        x, _ = data
        optimizer.zero_grad()

        z_mean, z_logvar, x_prime = model(x)
        loss = model.elbo_loss(x, z_mean, z_logvar, x_prime)
        loss.backward()
        optimizer.step()
        print(loss)

        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100
            print(f"batch {i+1}, loss; {last_loss}")
            running_loss = 0

    return last_loss


def train() -> nn.Module:
    data_path = pathlib.Path("./data")

    loaders = process_data.setup_data_loaders(data_path)
    model = bVAE(1024, 128, beta=4)
    optimizer = torch.optim.Adam

    avg_train_losses = []
    avg_val_losses = []
    
    best_val_loss = 1_000_000.

    for epoch in range(HYPERPARAMETERS['epochs']):
        model.train(True)
        avg_loss = train_one_epoch(model, optimizer, loaders["train"])

        running_val_loss = 0
        model.eval()

        with torch.no_grad():
            for i, val_data in enumerate(loaders["val"]):
                val_inputs, val_labels = val_data
                val_outputs = model(val_inputs)
                val_loss = model.elbo_loss(val_inputs, *val_outputs)
                running_val_loss += val_loss

        avg_val_loss = running_val_loss / (i + 1)
        print(f"Epoch {epoch} --- train loss: {avg_loss}; val loss: {avg_val_loss} ---")

        avg_train_losses.append(avg_loss)
        avg_val_losses.append(avg_val_losses)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = f"model_epoch_{epoch}"
            torch.save(model.state_dict(), model_save_path)

    return model

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.min()) * 1 / (tensor.max() - tensor.min())


def test_convolutions():
    data = torch.load("./processed_data/train.pt")
    test_image = data[0][0]
    m = nn.Sequential(
        nn.Conv2d(3, 48, 11, stride=4),
        nn.Conv2d(48, 24, 7, stride=2),
    )

    convolved = m(test_image)

    print(test_image.size())
    print(convolved.size())

    rows, cols = 4, 6
    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for i in range(1, 24 + 1):
        fig.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(denormalize(convolved[i - 1, :, :].detach().numpy()))

    plt.show()


def main() -> int:
    train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
