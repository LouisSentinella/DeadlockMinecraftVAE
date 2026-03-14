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

PERCEPTUAL_WEIGHT = config['perceptual_weight']
KL_WEIGHT = config['kl_weight']
CLASSIFIER_WEIGHT = config['classifier_weight']
Z_DIM = config['z_dim']
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


def loss_function_perceptual(recon_x, x, mu, logvar, kl_weight, perceptual_loss_module, class_logits, labels):
    p_loss = perceptual_loss_module(recon_x, x)

    mse_loss = F.mse_loss(recon_x, x)

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # normalise weights
    p_mult = 1
    mse_mult = 200

    p_loss = p_loss * p_mult
    mse_loss = mse_loss * mse_mult
    class_loss = F.cross_entropy(class_logits, labels.to(device), label_smoothing=0.2)

    total_recon_loss = PERCEPTUAL_WEIGHT * p_loss + (1 - PERCEPTUAL_WEIGHT) * mse_loss
    total_loss = total_recon_loss + kl_weight * kl_loss + CLASSIFIER_WEIGHT * class_loss

    return total_loss, total_recon_loss, kl_loss, class_loss

def train():
    dataset = GameScreenshotsDataset({"deadlock": DEADLOCK_DATA_PATH, "minecraft": MINECRAFT_DATA_PATH})
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = VAE(Z_DIM).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX)

    start_epoch = 0
    if config.get('checkpoint'):
        checkpoint = torch.load(config['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    perceptual_loss = PerceptualLoss().to(device)

    loss_log = []
    fixed_batch, _ = next(iter(dataloader))
    for epoch in range(start_epoch, EPOCHS):
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_loss = 0
        epoch_class_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch, labels in pbar:
            optimizer.zero_grad()

            recon_x, mu, logvar, classifier_logits = model(batch.to(device))

            loss, recon_loss, kl_loss, class_loss = loss_function_perceptual(recon_x, batch.to(device), mu, logvar, min(KL_WEIGHT, (epoch / 20) * KL_WEIGHT), perceptual_loss, classifier_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'kl': f"{kl_loss.item():.2f}",
                'recon_loss': f"{recon_loss.item():.4f}",
                'class_loss': f"{class_loss.item():.4f}"
            })

            optimizer.step()

            loss_log.append(loss.item())
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_class_loss += class_loss.item()
        epoch_loss /= len(dataloader)
        epoch_recon_loss /= len(dataloader)
        epoch_kl_loss /= len(dataloader)
        epoch_class_loss /= len(dataloader)
        print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Recon: {epoch_recon_loss:.4f} | KL: {epoch_kl_loss:.4f} | Class Loss: {epoch_class_loss:.4f}")
        scheduler.step()
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
            }, f"../checkpoints/epoch_{epoch}.pt")

            model.eval()
            with torch.no_grad():
                recon, _, _, _ = model(fixed_batch.to(device))

            originals = 0.5 * fixed_batch + 0.5
            reconstructions = 0.5 * recon.cpu() + 0.5

            comparison = torch.stack([originals, reconstructions], dim=1)
            comparison = comparison.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)

            grid = torchvision.utils.make_grid(comparison, nrow=8)
            torchvision.utils.save_image(grid, f"../outputs/recon_epoch_{epoch}.png")
            model.train()

if __name__ == '__main__':
    train()