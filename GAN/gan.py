import torch
from torch import nn, optim
from torch.autograd import Variable

CONFIG = {
    "logging_interval": 10
}


class GAN:
    def __init__(self, latent_dim, device, discriminator_lr=0.0002, generator_lr=0.0002):
        self.latent_dim = latent_dim
        self.device = device
        # Initializing the generator and discriminator networks
        self.GeneratorNet = Generator(device=self.device, latent_dim=self.latent_dim)
        self.DiscriminatorNet = Discriminator(device=self.device)
        # Initializing their optimizers
        self.discriminator_optimizer = optim.Adam(self.DiscriminatorNet.parameters(), lr=discriminator_lr)
        self.generator_optimizer = optim.Adam(self.GeneratorNet.parameters(), lr=generator_lr)
        # Defining the loss
        self.loss = nn.BCELoss()
        # Config
        self.config = CONFIG
        # Loss History
        self.history = {"Generator": [], "Discriminator": []}

    def sample_noise(self, size):
        return torch.randn(size, self.latent_dim)

    def train(self, X, discriminator_iterations=1, generator_iterations=1, batch_size=32, epochs=10):
        for i in range(epochs):
            mean_d_loss = 0
            mean_g_loss = 0
            k = 0
            for idx, (batch, _) in enumerate(X):
                # Train the discriminator
                noise = self.sample_noise(batch_size)
                d_loss = self.train_discriminator(batch, noise, discriminator_iterations)
                # Train the generator
                noise = self.sample_noise(batch_size)
                g_loss = self.train_generator(noise, generator_iterations)

                mean_d_loss += d_loss.item()
                mean_g_loss += g_loss.item()

                k += 1

            print(
                f"Epoch {i}: Discriminator loss: {mean_d_loss / k} | Generator loss: {mean_g_loss / k}")
            self.history["Generator"].append(mean_g_loss / k)
            self.history["Discriminator"].append(mean_d_loss / k)

    def train_generator(self, noise, k):
        for _ in range(k):
            # Reset gradients
            self.generator_optimizer.zero_grad()

            # Get D(G(z))
            prediction = self.DiscriminatorNet(self.GeneratorNet(noise))

            # Calculate the loss, i.e. Minimize -log(D(G(z))
            error = self.loss(prediction, Variable(torch.ones_like(prediction)))
            error.backward()

            # Update weights
            self.generator_optimizer.step()
            return error

    def train_discriminator(self, real, noise, k):
        for _ in range(k):
            # Reset gradients
            self.discriminator_optimizer.zero_grad()

            # Get D(real) and D(G(noise))
            prediction_real = self.DiscriminatorNet(real)
            with torch.no_grad():
                detached_fake = self.GeneratorNet(noise)
            prediction_fake = self.DiscriminatorNet(detached_fake)

            # Calculate the loss, i.e. Minimize -(log(D(real))+log(1-D(G(noise)))
            error_real = self.loss(prediction_real, Variable(torch.ones_like(prediction_real)))
            error_fake = self.loss(prediction_fake, Variable(torch.zeros_like(prediction_fake)))
            error = (error_fake + error_real) / 2
            error.backward()

            # Update weights
            self.discriminator_optimizer.step()
            return error

    def generate(self, size):
        '''Generate a sample.'''
        noise = self.sample_noise(size)
        return self.GeneratorNet(noise)


class Generator(nn.Module):
    def __init__(self, device, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        # Define the layers below
        self.hidden0 = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.hidden0(x)
        output = self.hidden1(output)
        output = self.hidden2(output)
        output = self.out(output)
        return output.to(self.device)


class Discriminator(nn.Module):
    def __init__(self, device):
        super(Discriminator, self).__init__()
        self.device = device
        input_dim = 784
        # Define the layers below
        self.hidden0 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.hidden0(x)
        output = self.hidden1(output)
        output = self.hidden2(output)
        output = self.out(output)
        return output.to(self.device)
