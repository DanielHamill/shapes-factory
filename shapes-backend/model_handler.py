import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from PIL import Image
import torch
from torch import nn
import numpy as np
from io import BytesIO
import base64
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchmetrics.classification import Accuracy

# geometric shapes classifier from: https://huggingface.co/prithivMLmods/Geometric-Shapes-Classification
labels = [
    "Circle ◯",
    "Kite ⬰",
    "Parallelogram ▰",
    "Rectangle ▭",
    "Rhombus ◆",
    "Square ◼",
    "Trapezoid ⏢",
    "Triangle ▲",
]


class PerfectModel:

    def __init__(self):
        print("Initializing Perfect Model.")
        model_name = "prithivMLmods/Geometric-Shapes-Classification"
        self.model = SiglipForImageClassification.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        print("Perfect Model ready.")

    def classify(self, image):
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        # predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
        predictions = {
            "label": labels[np.argmax(probs)],
            "confidence": float(np.max(probs)),
        }
        return predictions


class DenseModel(L.LightningModule):

    def __init__(self, image_size, hidden_layer1_size, hidden_layer2_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size, hidden_layer1_size),
            nn.ReLU(), 
            nn.Dropout(p=0.2),
            # nn.Linear(hidden_layer1_size, hidden_layer2_size),
            # nn.ReLU(),
            # nn.Dropout(p=0.2),
            # nn.Linear(hidden_layer2_size, output_size),
            nn.Linear(hidden_layer1_size, output_size),

        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.acc(y_hat, y)

        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True) # Logs average loss at the end of each epoch
        self.log("train_acc_epoch", acc, on_step=False, on_epoch=True) # Logs average loss at the end of each epoch
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.acc(y_hat, y)

        self.log("val_loss_epoch", loss, on_step=False, on_epoch=True) # Logs average loss at the end of each epoch
        self.log("val_acc_epoch", acc, on_step=False, on_epoch=True) # Logs average loss at the end of each epoch
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def _get_batch_from_image(self, image, category, n=100):
        image_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomInvert(p=1.0),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=(1, 1.25),
                shear=3,
                fill=255,
            ),
            transforms.Resize((20, 20)),
            transforms.ToTensor(),
        ])
        x = torch.stack([image_transforms(image) for _ in range(n)])
        indices = torch.full((n,), c, dtype=torch.long)
        y = torch.nn.functional.one_hot(indices, num_classes=2)
        return x, y

    def online_train(self, image, category):
        batch = self._get_batch_from_image(image, category)
        optimizer = self.configure_optimizers()
        loss = self.training_step(batch, 0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        


def test_classify():
    perfect_model = PerfectModel()
    # Image.fromarray(image).convert("RGB")
    # triangle_image = Image.open("./images/triangle.PNG")
    with open("./images/rectangle_b64.txt", "r") as f:
        base64_str = f.read()
    image_data = base64.b64decode(base64_str)
    rectangle_image = Image.open(BytesIO(image_data))

    print(perfect_model.classify(rectangle_image))


def test_resize():
    images_dir = "/home/danielhamill/Documents/projects/shapes-app/shapes-backend/images/temp"
    for i, filename in enumerate(os.listdir(images_dir)):
        image_path = f"{images_dir}/{filename}"
        test_image = Image.open(image_path)
        resized = test_image.resize((20, 20))
        resized.save(f"./images/temp/resized{i}.png")


def test_train_model():
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomInvert(p=1.0),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2),
            scale=(1, 1.25),
            shear=3,
            fill=255,
        ),
        transforms.Resize((20, 20)),                # Resize to fixed size
        transforms.ToTensor(),                        # Converts to tensor (shape: [1, H, W])
        # transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize for 1 channel
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomInvert(p=1.0),
        transforms.Resize((20, 20)),                # Resize to fixed size
        transforms.ToTensor(),                        # Converts to tensor (shape: [1, H, W])
        # transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize for 1 channel
    ])

    train_dataset = datasets.ImageFolder(root="./images/shapes_dataset/train", transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3)

    val_dataset = datasets.ImageFolder(root="./images/shapes_dataset/test", transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=3)

    model = DenseModel(image_size=20, hidden_layer1_size=200, hidden_layer2_size=50, output_size=3)

    mlf_logger = MLFlowLogger(
        experiment_name="lightning_logs",
        tracking_uri="file:./ml-runs",
        checkpoint_path_prefix="my_prefix"
    )

    trainer = L.Trainer(max_epochs=500, logger=mlf_logger)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # for i, (x, y) in enumerate(train_dataloader):
        # save_image(x, f"./images/temp/image{i}.png")
        # print(x)

if __name__ == "__main__":
    test_train_model()