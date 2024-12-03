import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class ASLDataset(Dataset):
    def __init__(self, data_dir, class_names, transform=None, samples_per_class=None, split='train',
                 validation_split=0.2):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        if split == 'test':
            # Handle test directory structure differently
            for class_idx, class_name in enumerate(class_names):
                if class_name == 'del':
                    # Skip 'del' class for test set since image doesn't exist
                    continue
                # Look for {class_name}_test.jpg file
                test_image = f"{class_name}_test.jpg"
                img_path = os.path.join(data_dir, test_image)

                if os.path.exists(img_path):
                    self.samples.append((img_path, class_idx))
                else:
                    raise FileNotFoundError(f"Test image not found for class {class_name}: {img_path}")
        else:
            # Original logic for train/val splits
            for class_idx, class_name in enumerate(class_names):
                class_path = os.path.join(data_dir, class_name)
                image_files = sorted([f for f in os.listdir(class_path) if f.endswith('.jpg')])

                if samples_per_class:
                    image_files = image_files[:samples_per_class]

                # Calculate split indices
                total_samples = len(image_files)
                val_size = int(total_samples * validation_split)

                if split == 'train':
                    image_files = image_files[val_size:]
                elif split == 'val':
                    image_files = image_files[:val_size]

                self.samples.extend([(os.path.join(class_path, img), class_idx)
                                     for img in image_files])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")

            image = cv2.resize(image, (200, 200))
            image = image / 255.0

            if self.transform:
                image = Image.fromarray((image * 255).astype(np.uint8))
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            # Return a default value or raise the exception
            raise e
