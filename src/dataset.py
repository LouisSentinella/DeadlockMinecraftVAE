import os

import PIL
from torch.utils.data import Dataset
import torchvision.transforms as T


class GameScreenshotsDataset(Dataset):
    def __init__(self, game_dict, transform=None,):
        self.data = []
        for i, game in enumerate(game_dict.values()):
            for img in os.listdir(game):
                self.data.append((os.path.join(game, img), i))

        self.transform = transform or T.Compose([T.Resize((360,400)), T.CenterCrop((256, 256)), T.Resize(64), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image PIL
        img = PIL.Image.open(self.data[idx][0])

        # Apply transforms (Center crop)
        img = self.transform(img)

        # Return (tensor, label)
        return img, self.data[idx][1]