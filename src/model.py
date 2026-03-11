import torch
import math
import torch.nn.functional as F
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=4, padding=1,  stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,kernel_size=4, padding=1,  stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.ConvBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,kernel_size=4, padding=1,  stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1,  stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        self.fc_mu = nn.Linear(in_features=2048, out_features=z_dim)
        self.fc_logvar = nn.Linear(in_features=2048, out_features=z_dim)

    def forward(self, x):
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = x.view(x.size(0), -1)

        return self.fc_mu(x), torch.clamp(self.fc_logvar(x), -2, 2)

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(in_features=z_dim, out_features=2048)
        self.UpsampleBlock1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.UpsampleBlock2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.UpsampleBlock3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.UpsampleBlock4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.UpsampleBlock5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 2, 2)
        x = self.UpsampleBlock1(x)
        x = self.UpsampleBlock2(x)
        x = self.UpsampleBlock3(x)
        x = self.UpsampleBlock4(x)
        x = self.UpsampleBlock5(x)
        return x

class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.Encoder = Encoder(z_dim)
        self.Decoder = Decoder(z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.Encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.Decoder(z)
        return reconstruction, mu, logvar
