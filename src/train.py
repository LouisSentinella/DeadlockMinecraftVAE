import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import PerceptualLoss

from src.dataset import GameScreenshotsDataset
from src.model import VAE

with open('../config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

ALPHA = config['alpha']
Z_DIM = config['z_dim']
BETA = config['beta']
LR = config['lr']
T_MAX = config['t_max']
BATCH_SIZE = config['batch_size']
EPOCHS = config['epochs']
IMAGE_SIZE = config['image_size']
DEADLOCK_DATA_PATH = config['data']['deadlock']
MINECRAFT_DATA_PATH = config['data']['minecraft']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

def loss_function(recon_x, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / recon_x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def loss_function_perceptual(recon_x, x, mu, logvar, beta, perceptual_loss_module):
    p_loss = perceptual_loss_module(recon_x, x)

    mse_loss = F.mse_loss(recon_x, x)

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    total_recon_loss = ALPHA * p_loss + (1 - ALPHA) * mse_loss
    total_loss = total_recon_loss + beta * kl_loss

    return total_loss, total_recon_loss, kl_loss

def train():
    dataset = GameScreenshotsDataset({"deadlock": DEADLOCK_DATA_PATH, "minecraft": MINECRAFT_DATA_PATH})
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = VAE(Z_DIM).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX)

    perceptual_loss = PerceptualLoss().to(device)

    loss_log = []
    fixed_batch, _ = next(iter(dataloader))
    for epoch in range(EPOCHS):
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch, labels in pbar:
            optimizer.zero_grad()

            recon_x, mu, logvar = model(batch.to(device))

            loss, recon_loss, kl_loss = loss_function_perceptual(recon_x, batch.to(device), mu, logvar, min(BETA, (epoch / 20) * BETA), perceptual_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'kl': f"{kl_loss.item():.2f}"
            })

            optimizer.step()

            loss_log.append(loss.item())
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
        epoch_loss /= len(dataloader)
        epoch_recon_loss /= len(dataloader)
        epoch_kl_loss /= len(dataloader)
        print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Recon: {epoch_recon_loss:.4f} | KL: {epoch_kl_loss:.4f}")
        scheduler.step()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"../checkpoints/epoch_{epoch}.pt")

            model.eval()
            with torch.no_grad():
                recon, _, _ = model(fixed_batch.to(device))

            originals = 0.5 * fixed_batch + 0.5
            reconstructions = 0.5 * recon.cpu() + 0.5

            comparison = torch.stack([originals, reconstructions], dim=1)
            comparison = comparison.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)

            grid = torchvision.utils.make_grid(comparison, nrow=8)
            torchvision.utils.save_image(grid, f"../outputs/recon_epoch_{epoch}.png")
            model.train()

if __name__ == '__main__':
    train()