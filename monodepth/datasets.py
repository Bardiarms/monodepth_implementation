import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class KITTIDataset(Dataset):
    """
    Loads stereo image pairs and bf values from a KITTI list file.
    Each line in list_file must be: left_img_path right_img_path bf_value
    """
    def __init__(self, list_file, height=256, width=512, training=False):
        super().__init__()
        with open(list_file, 'r') as f:
            self.samples = [line.strip().split() for line in f if line.strip()]
        self.height = height
        self.width = width
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        left_path, right_path, bf_str = self.samples[idx]
        left_img = Image.open(left_path).convert("RGB")
        right_img = Image.open(right_path).convert("RGB")

        # Resize
        left_img = TF.resize(left_img, (self.height, self.width))
        right_img = TF.resize(right_img, (self.height, self.width))

        # Convert to [0,1] tensors
        left_tensor = TF.to_tensor(left_img)
        right_tensor = TF.to_tensor(right_img)

        bf = torch.tensor(float(bf_str), dtype=torch.float32)

        return {
            "left": left_tensor,
            "right": right_tensor,
            "bf": bf,
            "left_path": left_path,
            "right_path": right_path,
        }
