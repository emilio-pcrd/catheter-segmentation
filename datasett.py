import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def arrange_dataset(dataset_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "images"))
        os.makedirs(os.path.join(output_dir, "masks"))

    img_counter = 1
    mask_counter = 1

    for idx, sub_dirs in enumerate(os.listdir(dataset_path)):
        for subdir in os.listdir(os.path.join(dataset_path, sub_dirs)):
            if subdir == "images":
                for img in sorted(
                    os.listdir(os.path.join(dataset_path, sub_dirs, subdir))
                ):
                    if img.endswith(".png") or img.endswith(".jpg"):
                        new_img_name = (
                            f"image_{idx}_{img_counter}{os.path.splitext(img)[1]}"
                        )
                        img_path = os.path.join(dataset_path, sub_dirs, subdir, img)
                        img_rgb = Image.open(img_path).convert("RGB")
                        img_rgb.save(os.path.join(output_dir, "images", new_img_name))
                        img_counter += 1
            else:
                assert subdir == "masks"
                for mask in sorted(
                    os.listdir(os.path.join(dataset_path, sub_dirs, subdir))
                ):
                    if mask.endswith(".jpg") or mask.endswith(".png"):
                        new_mask_name = (
                            f"mask_{idx}_{mask_counter}{os.path.splitext(mask)[1]}"
                        )
                        mask_path = os.path.join(dataset_path, sub_dirs, subdir, mask)
                        mask_ = Image.open(mask_path).convert("L")
                        mask_.save(os.path.join(output_dir, "masks", new_mask_name))
                        mask_counter += 1


class CatheterDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        super().__init__()
        self.img_path = os.path.join(dataset_path, "images")
        self.mask_path = os.path.join(dataset_path, "masks")

        self.imgs = sorted(os.listdir(self.img_path))
        self.masks = sorted(os.listdir(self.mask_path))

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.imgs[index]))
        mask = Image.open(os.path.join(self.mask_path, self.masks[index]))
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return {"image": img, "label": mask.squeeze()}


class SAMDataset(Dataset):
    def __init__(self, dataset, processor, transform=None):
        self.dataset = dataset
        self.processor = processor
        if transform:
            self.transform = transform
        else:
            self.transform = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        # get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        image = np.clip(image, None, 1.0)
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask
        if self.transform:
            inputs = self.transform(inputs)
        return inputs


def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    if len(ground_truth_map.shape) != 2:
        raise ValueError(f"Expected a 2D mask, but got shape {ground_truth_map.shape}")
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox
