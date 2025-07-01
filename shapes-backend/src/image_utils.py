from torchvision import transforms
from torchvision import datasets, transforms
from typing import Any, Dict, List
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
import numpy as np
from scipy import ndimage
from PIL import Image

class CenterGreyscaleImage(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def pad_and_crop_around_center(self, image, center_point, pad_value=0):
       height, width = image.shape[:2]
       x, y = center_point

       # Calculate padding amounts
       pad_left = max(0, width // 2 - x)
       pad_right = max(0, x + width // 2 - width)
       pad_top = max(0, height // 2 - y)
       pad_bottom = max(0, y + height // 2 - height)

       # Pad the image
       padded_image = np.pad(image,
                            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))[:image.ndim],
                            mode='constant', constant_values=pad_value)

       # Calculate new center coordinates after padding
       new_x = x + pad_left
       new_y = y + pad_top

       # Calculate crop boundaries
       start_x = new_x - width // 2
       end_x = new_x + width // 2
       start_y = new_y - height // 2
       end_y = new_y + height // 2

       # Crop the image
       cropped_image = padded_image[start_y:end_y, start_x:end_x]

       return cropped_image

    def calc_center(self, img: Image):
        y, x = ndimage.center_of_mass(img)
        return int(x), int(y)

    def forward(self, img: Image):
        cx, cy = self.calc_center(img)
        cropped = self.pad_and_crop_around_center(np.array(img), (cx, cy))
        return Image.fromarray(cropped)
    

model_transforms = {
    "train": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomInvert(p=1.0),
        CenterGreyscaleImage(),
        transforms.RandomAffine(
            degrees=0,
            # translate=(0.2, 0.2),
            scale=(0.8, 1.2),
            shear=3,
            fill=0,
        ),
        transforms.Resize((20, 20)),                # Resize to fixed size
        transforms.ToTensor(),                        # Converts to tensor (shape: [1, H, W])
    ]),

    "val": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomInvert(p=1.0),
        CenterGreyscaleImage(),
        transforms.Resize((20, 20)),                # Resize to fixed size
        transforms.ToTensor(),                        # Converts        to tensor (shape: [1, H, W])
    ])
}

def get_batch_from_images(images: list, category, transforms=model_transforms["train"], n=100):
    """Get a batch of images with random transforms from a list of images."""
    batch_per_image = n // len(images)
    x = torch.stack([transforms(image) for _ in range(batch_per_image) for image in images])
    y = torch.full((len(x),), category, dtype=torch.long)
    return x, y