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

        for param in self.model.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, recon_x, x):
        recon_x = (recon_x + 1) / 2
        x = (x + 1) / 2

        recon_x = (recon_x - self.mean) / self.std
        x = (x - self.mean) / self.std

        perceptual_loss = self.loss(self.model(x), self.model(recon_x))

        return perceptual_loss