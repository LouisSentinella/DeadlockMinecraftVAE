import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from src.model import VAE
from src.dataset import GameScreenshotsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('../config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

Z_DIM = config['z_dim']
BATCH_SIZE = config['batch_size']
DEADLOCK_DATA_PATH = config['data']['deadlock']
MINECRAFT_DATA_PATH = config['data']['minecraft']


def compute_centroids(dataLoader, model):
    deadlock_mu = 0
    mincraft_mu = 0
    deadlock_count = 0
    minecraft_count = 0
    for batch, labels in dataLoader:
        mu, logvar = model.Encoder(batch.to(device))

        for index, label in enumerate(labels):
            if label == 0:
                deadlock_mu += mu[index]
                deadlock_count += 1
            else:
                mincraft_mu += mu[index]
                minecraft_count += 1

    deadlock_mu /= deadlock_count
    mincraft_mu /= minecraft_count

    return deadlock_mu, mincraft_mu

def interpolate(z1, z2, model, n):
    z1_mu, _ = model.Encoder(z1.unsqueeze(0).to(device))
    z2_mu, _ = model.Encoder(z2.unsqueeze(0).to(device))

    decoded_frames = []

    for alpha in torch.linspace(0, 1, n):
        z = (1 - alpha) * z1_mu + alpha * z2_mu
        decoded = model.Decoder(z.unsqueeze(0))
        decoded = (decoded.cpu() * 0.5 + 0.5)

        decoded_frames.append(decoded)

    comparison = torch.stack(decoded_frames, dim=1)
    comparison = comparison.view(-1, 3, 64, 64)

    grid = torchvision.utils.make_grid(comparison, nrow=8)

    torchvision.utils.save_image(grid, f"../outputs/interpolate.png")


def style_transfer(image, source_centroid, target_centroid, model, n=10):
    mu, _ = model.Encoder(image.unsqueeze(0).to(device))

    direction = target_centroid - source_centroid

    frames = []
    for alpha in torch.linspace(0, 1, n):
        z = mu + alpha * direction
        decoded = model.Decoder(z)
        decoded = decoded.cpu() * 0.5 + 0.5
        frames.append(decoded)

    comparison = torch.cat(frames, dim=0)
    grid = torchvision.utils.make_grid(comparison, nrow=n)
    torchvision.utils.save_image(grid, f"../outputs/style_transfer.png")

if __name__ == '__main__':
    dataset = GameScreenshotsDataset({"deadlock": DEADLOCK_DATA_PATH, "minecraft": MINECRAFT_DATA_PATH})
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = VAE(Z_DIM).to(device)
    model.load_state_dict(torch.load("../checkpoints/64perceptual05mse/epoch_99.pt"))
    model.eval()

    deadlock_centroid, minecraft_centroid = compute_centroids(dataloader, model)

    deadlock_img, _ = dataset[500]
    minecraft_img, _ = dataset[-500]

    interpolate(deadlock_img, minecraft_img, model, 8)

    style_transfer(deadlock_img, deadlock_centroid, minecraft_centroid, model)