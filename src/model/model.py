#!/usr/bin/env python
import pathlib
import random
import sys
from typing import Any, Type

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import process_data

HYPERPARAMETERS = {
    "batch_size": 128,
    "learning_rate": 1.0e-3,
    "use_cuda": False,
    "epochs": 10,
}

LABEL_MAP = {0: "Glaucoma not Present", 1: "Glaucoma Present"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Encoder(nn.Module):
    def __init__(self, hidden_size, latent_size) -> None:
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc_mu = nn.Linear(128, latent_size)
        self.fc_logvar = nn.Linear(128, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, hidden_size, latent_size) -> None:
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_size, 32 * 56 * 56)
        self.deconv1 = nn.ConvTranspose2d(
            32, 16, 3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            16, 3, 3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), 32, 56, 56)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x


class bVAE(nn.Module):
    def __init__(self, hidden_size, latent_size, beta=1) -> None:
        super(bVAE, self).__init__()
        self.z_dim = latent_size
        self.encoder = Encoder(hidden_size, latent_size)
        self.decoder = Decoder(hidden_size, latent_size)
        self.beta = beta

    def elbo_loss(self, x, z_mean, z_logvar, x_prime):
        reconstruction_loss = F.mse_loss(x, x_prime)
        kld = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        kld_loss = kld.mean()
        return reconstruction_loss + self.beta * kld_loss

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)

        # Sample with reparametrization trick
        z = distributions.Normal(z_mean, torch.exp(z_logvar / 2)).rsample((1,))  # type: ignore
        z = z.reshape((z.shape[1], self.z_dim))

        x_prime = self.decoder(z)
        return z_mean, z_logvar, x_prime

    def sample(self, mean, logvar):
        z = distributions.Normal(mean, torch.exp(logvar / 2)).sample((1,))  # type: ignore
        return self.decoder(z)

    def infer(self, x):
        z_mean, z_logvar = self.encoder(x)
        return distributions.Normal(z_mean, torch.exp(z_logvar / 2)).rsample((1,))


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
        print(
            f"\033[2KBatch: {i}"  # ]
            f"〔\033[32m{u'━' * (i + 1)}{' ' * ((len(train_loader) // HYPERPARAMETERS['batch_size']) - i)}\033[0m〕"  # ]]
            f"loss: {loss}",
            end="\r",
            flush=True,
        )
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100
            # print(f"batch {i+1}, loss; {last_loss}")
            running_loss = 0

    print()
    return last_loss


def setup_data_loaders(data: pathlib.Path) -> dict[str, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.RandomAutocontrast(1),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    train_dataset = datasets.ImageFolder(str(data / "train"), transform=transform)
    test_dataset = datasets.ImageFolder(str(data / "test"), transform=transform)
    val_dataset = datasets.ImageFolder(str(data / "val"), transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=True
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def train() -> nn.Module:
    data_path = pathlib.Path("./data")

    loaders = setup_data_loaders(data_path)
    model = bVAE(1024, 128, beta=4)
    optimizer = torch.optim.Adam

    avg_train_losses = []
    avg_val_losses = []

    best_val_loss = 1_000_000.0

    for epoch in range(HYPERPARAMETERS["epochs"]):
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
        nn.Conv2d(3, 16, 3, stride=2, padding=1),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        # nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        # nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)
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
        # plt.imshow(denormalize(
        # np.transpose(convolved.detach().numpy(), (1,2,0))
        # ))
        plt.imshow(denormalize(convolved[i - 1, :, :].detach().numpy()))

    plt.show()


def image_reconstruction():
    data_path = pathlib.Path("./data")
    model = bVAE(1024, 128, beta=4)
    model.load_state_dict(torch.load("./archive/v0.1/model_epoch_9"))
    loaders = setup_data_loaders(data_path)

    images, labels = next(iter(loaders["train"]))
    z_mean, z_logvar, reconstructed = model(images)
    rows, cols = 6, 2
    fig = plt.figure(figsize=(6, 18))
    fig.subplots_adjust(hspace=0.5)
    for i in range(1, rows * cols + 1, 2):
        sample_idx = random.randint(0, HYPERPARAMETERS["batch_size"])
        img, label = images[sample_idx], labels[sample_idx]
        reconstructed_img = reconstructed[sample_idx]

        img = np.transpose(denormalize(img).detach(), (1, 2, 0))
        fig.add_subplot(rows, cols, i)
        plt.title(f"Image {i // 2 + 1}\n{LABEL_MAP[label.item()]}")
        plt.axis("off")
        plt.imshow(img)

        reconstructed_img = np.transpose(
            denormalize(reconstructed_img).detach(), (1, 2, 0)
        )
        fig.add_subplot(rows, cols, i + 1)
        plt.title(f"Reconstruction of image {i // 2 + 1}")
        plt.axis("off")
        plt.imshow(reconstructed_img)
    plt.show()


def dimensionality_reduction():
    data_path = pathlib.Path("./data")
    model = bVAE(1024, 128, beta=4)
    model.load_state_dict(torch.load("./archive/v0.1/model_epoch_9"))
    loaders = setup_data_loaders(data_path)

    latent_space = []
    ground_truth = []
    for i, (images, labels) in enumerate(loaders["val"]):
        print(f"Processing batch: {i}")
        latent_space.append(model.infer(images)[0])
        ground_truth.append(labels)

    latent_space = torch.concat(latent_space).detach().numpy()
    ground_truth = torch.concat(ground_truth).detach().numpy()

    pca_reducer = PCA()
    pca_embedding = pca_reducer.fit_transform(latent_space)

    tsne_reducer = TSNE()
    tsne_embedding = tsne_reducer.fit_transform(latent_space)

    umap_reducer = UMAP()
    umap_embedding = umap_reducer.fit_transform(latent_space)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,6))
    for i, embedding in enumerate([pca_embedding, tsne_embedding, umap_embedding]):
        sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=ground_truth, ax=ax[i])
        handles, _ = ax[i].get_legend_handles_labels()
        ax[i].legend(handles, LABEL_MAP.values())

    ax[0].set_title("PCA")
    ax[1].set_title("t-SNE")
    ax[2].set_title("UMAP")
    plt.show()



def main() -> int:
    dimensionality_reduction()
    return 0


if __name__ == "__main__":
    sys.exit(main())
