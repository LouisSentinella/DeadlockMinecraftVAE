from torch import nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import Normalize
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.loss = nn.MSELoss()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        self.model = vgg.eval()

        self.slice1 = nn.Sequential(*self.model[:4])  # Output of conv1_2
        self.slice2 = nn.Sequential(*self.model[4:9])  # Output of conv2_2
        self.slice3 = nn.Sequential(*self.model[9:16])  # Output of conv3_3

        for param in self.model.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, recon_x, x):
        recon_x = (recon_x + 1) / 2
        x = (x + 1) / 2

        recon_x = (recon_x - self.mean) / self.std
        x = (x - self.mean) / self.std

        # Get features from multiple depths
        h1_x = self.slice1(x)
        h1_rx = self.slice1(recon_x)

        h2_x = self.slice2(h1_x)
        h2_rx = self.slice2(h1_rx)

        h3_x = self.slice3(h2_x)
        h3_rx = self.slice3(h2_rx)

        l1 = self.loss(h1_rx, h1_x)
        l2 = self.loss(h2_rx, h2_x)
        l3 = self.loss(h3_rx, h3_x)

        perceptual_loss = l1 + l2 + l3

        return perceptual_loss