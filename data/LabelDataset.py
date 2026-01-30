import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = self._create_pairs()

    def _create_pairs(self):
        pairs = []
        folders = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f)) and f.isdigit()]

        for folder in folders:
            folder_path = os.path.join(self.root_dir, folder)
            files = os.listdir(folder_path)

            original_files = [f for f in files if f.endswith('.original')]
            reference_pngs = [f for f in files if f.startswith('reference') and f.endswith('.png')]
            parsing_mask_npy = [f for f in files if f == 'parsing_mask.npy']

            # ensure necessary files
            if len(original_files) == 1 and len(reference_pngs) == 1 and len(parsing_mask_npy) == 1:
                original_file = original_files[0]
                reference_png = reference_pngs[0]
                parsing_mask = parsing_mask_npy[0]

                pairs.append((folder_path, original_file, reference_png, parsing_mask, folder))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        folder_path, original_name, reference_name, parsing_mask_name, folder = self.pairs[idx]

        original_path = os.path.join(folder_path, original_name)
        reference_path = os.path.join(folder_path, reference_name)
        parsing_mask_path = os.path.join(folder_path, parsing_mask_name)

        original_img = Image.open(original_path).convert("RGB")
        reference_img = Image.open(reference_path).convert("RGB")

        parsing_mask = np.load(parsing_mask_path)

        if self.transform:
            original_img = self.transform(original_img)
            reference_img = self.transform(reference_img)

        # process parsing mask
        prompt_mask = np.ones_like(parsing_mask, dtype=bool)
        prompt_mask[np.isin(parsing_mask, [0, 4, 7, 8, 11])] = False # ['background', 'lip', 'eyebrows', 'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface', 'mouth', 'eye_glass', 'ear_rings']

        transformed_H, transformed_W = reference_img.shape[-2:]
        prompt_mask_image = Image.fromarray(prompt_mask.astype(np.uint8) * 255)
        prompt_mask_resized = prompt_mask_image.resize((transformed_W, transformed_H), Image.NEAREST)
        bool_parsing_mask_numpy = (np.array(prompt_mask_resized) > 0).astype(bool)

        return original_img, reference_img, bool_parsing_mask_numpy, folder