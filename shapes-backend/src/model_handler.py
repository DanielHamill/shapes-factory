import base64
import os
import random
import uuid
from io import BytesIO
from typing import Dict
import asyncio

import lightning as L
import mlflow
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, SiglipForImageClassification

from image_utils import model_transforms, get_batch_from_images

MODEL_LIFETIME = 10 * 60
# MODEL_LIFETIME = 10

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

    def __init__(
        self,
        image_size,
        hidden_layer1_size,
        output_size,
        additional_imgs_during_training=5,
        batch_size=500,
        num_batches=1,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size, hidden_layer1_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_layer1_size, output_size),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=output_size)
        self.automatic_optimization = False
        self.val_step_counter = 0
        self.seen = [[]] * output_size
        self.additional_imgs_during_training = additional_imgs_during_training
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_categories = output_size

        val_dataset = datasets.ImageFolder(
            root="./images/shapes_dataset/test", transform=model_transforms["val"]
        )
        self.val_dataset = DataLoader(
            val_dataset, batch_size=4, shuffle=False, num_workers=3
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.acc(y_hat, y)

        # self.log("train_loss_epoch", loss, on_step=False, on_epoch=True) # Logs average loss at the end of each epoch
        # self.log("train_acc_epoch", acc, on_step=False, on_epoch=True) # Logs average loss at the end of each epoch
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        mask_0 = y == 0
        mask_1 = y == 1

        y_hat_0 = y_hat[mask_0]
        y_hat_1 = y_hat[mask_1]
        y_0 = y[mask_0]
        y_1 = y[mask_1]

        acc = self.acc(y_hat, y)
        if len(y_0) != 0:
            acc0 = self.acc(y_hat_0, y_0)
        else:
            acc0 = np.nan

        if len(y_1) != 0:
            acc1 = self.acc(y_hat_1, y_1)
        else:
            acc1 = np.nan

        return loss, (acc, acc0, acc1)

    def run_validation(self, val_dataloader):
        self.eval()
        device = next(self.parameters()).device

        total_loss = []
        acc_all = []
        acc0_all = []
        acc1_all = []
        count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = tuple(t.to(device) for t in batch)
                loss, (acc, acc0, acc1) = self.validation_step(batch, batch_idx)
                total_loss.append(loss)
                acc_all.append(acc)
                acc0_all.append(acc0)
                acc1_all.append(acc1)
                count += 1

        avg_loss = np.mean(total_loss)

        avg_acc = np.mean(acc_all)
        avg_acc0 = np.nanmean(acc0_all)
        avg_acc1 = np.nanmean(acc1_all)

        # Log to MLflow
        mlflow.log_metric("val_loss", avg_loss, step=self.val_step_counter)
        mlflow.log_metric("val_accuracy", avg_acc, step=self.val_step_counter)
        self.val_step_counter += 1

        return avg_loss, (avg_acc, avg_acc0, avg_acc1)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.acc(y_hat, y)

        self.log(
            "test_loss_epoch", loss, on_step=False, on_epoch=True
        )  # Logs average loss at the end of each epoch
        self.log(
            "test_acc_epoch", acc, on_step=False, on_epoch=True
        )  # Logs average loss at the end of each epoch
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def _merge_batches(self, batches):
        xs, ys = zip(*batches)  # unzip into two lists
        merged_x = torch.cat(xs, dim=0)
        merged_y = torch.cat(ys, dim=0)
        return merged_x, merged_y

    def _online_training_step(self, batch):
        """Perform one online training step."""

        optimizer = self.configure_optimizers()
        loss = self.training_step(batch, 0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _get_images_for_category(self, category, n):
        """Get additional images from category that have been seen before to assist with training."""

        if len(self.seen[category]) <= n:
            return self.seen[category]
        return random.sample(self.seen[category], k=n)
    


    def train_online(self, image_b64, category):
        """Perform online training with an image."""
        num_batches = 1
        images_per_cat = self.additional_imgs_during_training // self.num_categories
        batch_size_per_category = (
            self.batch_size // self.num_categories
        )
        train_images = [
            self._get_images_for_category(category, images_per_cat)
            for category in range(self.num_categories)
        ]
        if len(train_images[category]) < images_per_cat:
            train_images[category].append(image_b64)
        else:
            train_images[category][0] = image_b64

        for _ in range(num_batches):
            batches = [
                get_batch_from_images(
                    [Image.open(BytesIO(base64.b64decode(img))) for img in cat_images],
                    cat,
                    n=batch_size_per_category,
                )
                for cat, cat_images in enumerate(train_images)
            ]
            batch = self._merge_batches(batches)
            self._online_training_step(batch)
        # self.run_validation(self.val_dataset)
        self.seen[category].append(image_b64)
        self.val_step_counter += 1


class ModelHandler:

    models: Dict[str, L.LightningModule]

    def __init__(self):
        # self.dense_model = DenseModel(
        #     image_size=20, hidden_layer1_size=200, output_size=2
        # )
        # self.perfect_model = PerfectModel()
        # self.model = self.dense_model
        print("Loaded all models.")
        self.models = {}

    def _get_model(self, user_id):
        if user_id in self.models:
            return self.models[user_id]
        else:
            print(f"Creating new model for user {user_id}")
            self.models[user_id] = DenseModel(
                image_size=20, hidden_layer1_size=200, output_size=3
            )
            asyncio.create_task(self.delete_model_after_delay(user_id, MODEL_LIFETIME))
            return self.models[user_id]

    async def delete_model_after_delay(self, user_id: str, delay: int):
        await asyncio.sleep(delay)
        if user_id in self.models:
            print(f"Deleting model for user {user_id}")
            del self.models[user_id]
        else:
            raise ValueError("Tried to delete a model that doesn't exist.")

    def predict(self, image_b64, user_id):
        encoded_string = base64.b64decode(image_b64)
        image_data = Image.open(BytesIO(encoded_string))
        model = self._get_model(user_id)
        with torch.no_grad():
            prediction = torch.softmax(
                model(model_transforms["val"](image_data))[0], dim=-1
            )
        label = torch.argmax(prediction).tolist()
        confidence = 200 * (float(torch.max(prediction)) - 0.33)
        # TODO: change confidence computation if not 2 classes

        # loss, (acc, acc0, acc1) = self.model.run_validation(self.model.val_dataset)
        # print("loss", loss)
        # print("accuracy", acc)
        # print("accuracy 0", acc0)
        # print("accuracy 1", acc1)
        print("prediction", prediction)
        return {
            "label": label,
            "confidence": confidence,
        }

    def benchmark(self, image_b64):
        encoded_string = base64.b64decode(image_b64)
        image_data = Image.open(BytesIO(encoded_string))
        # image_data.save(f"./images/temp/{uuid.uuid1()}.png")
        with torch.no_grad():
            prediction = torch.softmax(
                self.model(model_transforms["val"](image_data))[0], dim=-1
            )
        print(prediction)
        label = torch.argmax(prediction).tolist()
        print(label)
        # loss, acc = self.model.run_validation(self.model.val_dataset)
        # print("loss", loss)
        # print("accuracy", acc)
        # return {"message": "Hello World"}'
        return {"label": label}

    def save(self, image_b64):
        encoded_string = base64.b64decode(image_b64)
        image_data = Image.open(BytesIO(encoded_string))
        if not os.path.exists("./images/temp/"):
            os.mkdir("./images/temp/")
        image_data.save(f"./images/temp/{uuid.uuid1()}.png")
        return {}

    def train(self, image_b64, category, user_id):
        model = self._get_model(user_id)
        model.train_online(image_b64, category)
