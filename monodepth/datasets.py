import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import os

class KITTIDataset(Dataset):
    def __init__(self, list_file, data_root, height=256, width=512, training=False):
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.training = training

        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    raise ValueError(f"Line malformed: {line}")
                
                drive, left_rel, right_rel = parts
                left_path = os.path.join(data_root, drive, left_rel)
                right_path = os.path.join(data_root, drive, right_rel)

                # KITTI baseline × focal length (from P_rect_02)
                # = 0.54m × fx (≈ 721 px)
                bf_value = 0.54 * 721.0

                self.samples.append((left_path, right_path, bf_value))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        left_path, right_path, bf_value = self.samples[idx]

        left_img = Image.open(left_path).convert("RGB")
        right_img = Image.open(right_path).convert("RGB")

        left_img = TF.resize(left_img, (self.height, self.width))
        right_img = TF.resize(right_img, (self.height, self.width))

        left_tensor = TF.to_tensor(left_img)
        right_tensor = TF.to_tensor(right_img)

        bf = torch.tensor(bf_value, dtype=torch.float32)

        return {
            "left": left_tensor,
            "right": right_tensor,
            "bf": bf,
            "left_path": left_path,
            "right_path": right_path,
        }

