import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import matplotlib.pyplot as plt

torch.manual_seed(0)
plt.rcParams['figure.dpi'] = 200

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    print('WARNING: Using cpu, no gpu available')


class VariationalEncoder(nn.Module):
    """
    q_phi (z | x)
    """
    def __init__(self, latent_dim):
        super(VariationalEncoder, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self.fc3 = nn.Linear(512, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        mu = self.fc2(x)
        sigma = torch.exp(self.fc3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z


class VariationalDecoder(nn.Module):
    """
    p_theta (x | z)
    """
    def __init__(self, latent_dim):
        super(VariationalDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 784)

    def forward(self, z):
        z = self.fc1(z)
        z = F.relu(z)
        z = self.fc2(z)
        z = torch.sigmoid(z)
        return z.reshape((-1, 1, 28, 28))


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dim)
        self.decoder = VariationalDecoder(latent_dim)

    def forward(self, x):
        """
        Result of sampling for inference q(z|x), then
        getting the mean of p(x | z) for that sample.
        :param x:
        :return:
        """
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        print(f'Beginning epoch {epoch}')

        mean_loss = 0.0
        total_N = 0

        for x, y in data:
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()

            mean_loss += loss.detach()

        mean_loss /= total_N
        print(f'Mean loss: {loss}')
    return autoencoder


data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data',
                               transform=torchvision.transforms.ToTensor(),
                               download=True),
    batch_size=128, shuffle=True)
latent_dims = 2
vae = VariationalAutoencoder(latent_dims).to(device)
vae = train(vae, data)


def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        print('.')
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            plt.show()
            break


plot_latent(vae, data)

