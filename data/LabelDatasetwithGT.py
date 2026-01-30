import os
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import json

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

            original_files = [f for f in files if f.endswith('.jpg')]
            reference_pngs = [f for f in files if f.startswith('Reference') and f.endswith('.png')]
            parsing_mask_npy = [f for f in files if f.endwith('parsing_mask.npy')]
            json_files = [f for f in files if f.endswith('.json')] # json GT

            if len(original_files) == 1 and len(reference_pngs) == 1 and len(parsing_mask_npy) == 1 and len(json_files) == 1:
                original_file = original_files[0]
                reference_png = reference_pngs[0]
                parsing_mask = parsing_mask_npy[0]
                gt = json_files[0]

                pairs.append((folder_path, original_file, reference_png, parsing_mask, gt, folder))

        return pairs

    def __len__(self):
        return len(self.pairs)
    
    def json_to_mask(self, json_path, image_size):

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)

        for shape in data.get('shapes', []):
            points = shape.get('points', [])
            if points:
                flattened = [tuple(point) for point in points]
                draw.polygon(flattened, outline=1, fill=1)
        
        mask_np = np.array(mask)
        return mask_np

    def __getitem__(self, idx):
        folder_path, original_name, reference_name, parsing_mask_name, gt_name, folder = self.pairs[idx]

        original_path = os.path.join(folder_path, original_name)
        reference_path = os.path.join(folder_path, reference_name)
        parsing_mask_path = os.path.join(folder_path, parsing_mask_name)
        gt_path = os.path.join(folder_path, gt_name)

        original_img = Image.open(original_path).convert("RGB")
        reference_img = Image.open(reference_path).convert("RGB")

        parsing_mask = np.load(parsing_mask_path)

        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
            image_width = gt_data.get('imageWidth', original_img.width)
            image_height = gt_data.get('imageHeight', original_img.height)
            image_size = (image_width, image_height)
        
        gt_mask = self.json_to_mask(gt_path, image_size)

        if self.transform:
            original_img = self.transform(original_img)
            reference_img = self.transform(reference_img)

        prompt_mask = np.ones_like(parsing_mask, dtype=bool)
        prompt_mask[np.isin(parsing_mask, [0, 4, 7, 8, 11])] = False # ['background', 'lip', 'eyebrows', 'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface', 'mouth', 'eye_glass', 'ear_rings']

        if self.transform:
            _, transformed_H, transformed_W = original_img.shape
        else:
            transformed_H, transformed_W = reference_img.size 

        prompt_mask_image = Image.fromarray(prompt_mask.astype(np.uint8) * 255)
        prompt_mask_resized = prompt_mask_image.resize((transformed_W, transformed_H), Image.NEAREST)
        bool_parsing_mask_numpy = (np.array(prompt_mask_resized) > 0).astype(bool)

        if self.transform:
            gt_mask_image = Image.fromarray(gt_mask.astype(np.uint8) * 255)
            gt_mask_resized = gt_mask_image.resize((transformed_W, transformed_H), Image.NEAREST)
            gt_mask_numpy = (np.array(gt_mask_resized) > 0).astype(np.uint8)
        else:
            gt_mask_numpy = gt_mask

        return original_img, reference_img, bool_parsing_mask_numpy, gt_mask_numpy, folder