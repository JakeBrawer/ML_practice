import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from typing import Tuple
from matplotlib import pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_training_samples():
    """Visualizes a few training samples from the MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    fig, axes = plt.subplots(2, 5, figsize=(10, 2))
    for i in range(2):
        for j in range(5):
            img, label = mnist_train[i * 5 + j]
            print(f"Sample {i}: Label = {label}")
            axes[i, j] .imshow(img.squeeze(), cmap='gray')
            # axes[i,j].set_title(f'Label: {label}')
            # add space between rows
            axes[i,j].set_xticks([])

            axes[i,j].axis('off')

    plt.show()


class AutoEncoder(nn.Module):
    def __init__(self, d_in=784, dropout=.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(512, 64),
            nn.ReLU(),
            # nn.Dropout(dropout),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(512, d_in),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        output = self.encoder(x)
        # print(f"Output shape: {output.shape}")
        return self.decoder(output)


class VariationalAutoEncoder(nn.Module):
    """A Variational Autoencoder (VAE) implementation."""

    def __init__(self, d_in=784, latent_dim=64, dropout=.1):
        super().__init__()
        self._latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(512, latent_dim * 2)  # Output both mean and log variance
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(512, d_in),
            # nn.Dropout(dropout),
            nn.Sigmoid() # Output needs to be between 0 and 1 for Binary cross entropy
        )


    def reparameterize(self, mu, log_var):
        """Reparameterization trick to sample from the latent space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = (encoded[:, :self._latent_dim],
                      encoded[:, self._latent_dim:])

        z = self.reparameterize(mu, logvar)

        return self.decoder(z), mu, logvar


    def sample(self, n_samples=5, device=device):
        self.eval()

        with torch.no_grad():
            noise = torch.normal(0, 1,
                                 size=(n_samples,
                                       self._latent_dim)).to(device)
            return self.decoder(noise)


def visualize_reconstructions(original: torch.Tensor, reconstructed: torch.Tensor, n_samples: int = 10):
    """Visualizes original and reconstructed images."""
    original = original.view(-1, 1, 28, 28)
    reconstructed = reconstructed.view(-1, 1, 28, 28)
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 3))
    for i in range(n_samples):
        axes[0, i].imshow(original[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].axis('off')
    plt.show()





if __name__ == '__main__':
    # visualize_training_samples()
    # print("Training samples visualized successfully.")
    # train_autoencoder(device)
    train_vae(device=device)
