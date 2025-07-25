import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models.autoencoder import (VariationalAutoEncoder, AutoEncoder,
                                visualize_reconstructions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_autoencoder(device, batch_size=16, epochs=500, visualize_epoch=10):
    model = AutoEncoder()
    model.to(device)

    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root='./data', train=True,
                                 download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(mnist_train,
                                              batch_size=batch_size,
                                              shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for X, y in data_loader:

            X = X.squeeze(1).to(device)
            B, m,n = X.shape
            X = torch.reshape(X, (B, m*n))

            output = model(X)
            loss = criterion(output, X)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}\n")

        # Every visualize_epoch epochs we want to visualize the results of # the autoencoder
        # compared to the original images.
        if epoch % visualize_epoch == 0:
            model.eval()
            with torch.no_grad():
                X, y = next(iter(data_loader))
                X = X.squeeze(1).to(device)

                B, m, n = X.shape
                X = torch.reshape(X, (B, m * n))

                output = model(X)
                visualize_reconstructions(X, output, 10)

            model.train()


def train_vae(device, batch_size=32, n_samples=10, epochs=250, visualize_epoch=50):
    vae = VariationalAutoEncoder()
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root='./data', train=True,
                                 download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(mnist_train,
                                              batch_size=batch_size,
                                              shuffle=True)

    vae.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for X, y in data_loader:

            X = X.squeeze(1).to(device)
            B, m,n = X.shape
            X = torch.reshape(X, (B, m*n))

            reconstructed_X, mu, logvar = vae(X)

            ## Implement ELBO loss
            # First calculate reconstruction loss
            recon_loss = F.binary_cross_entropy(reconstructed_X, X,
                                                reduction='sum')
            # Then KL divergence
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon_loss + kl_div) / B

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}\n")

        # Every visualize_epochs epochs we want to visualize a few reconstructions
        # compared to gt
        n_per_row = n_samples // 2

        if epoch % visualize_epoch == 0:
            vae.eval()
            with torch.no_grad():
                output = vae.sample(n_samples, device)
                visualize_reconstructions(X, output, n_samples)

            vae.train()

if __name__ == '__main__':

    train_vae(device)

