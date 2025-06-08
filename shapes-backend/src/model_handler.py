import os
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
import torch
from torch import nn
import numpy as np
import lightning as L
from torchvision import transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
import mlflow
from io import BytesIO
import base64
from PIL import Image
import uuid
import random
from image_utils import model_transforms


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
            additional_imgs_during_training = 5,
            batch_size = 500,
            num_batches = 1,
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
        self.seen = [[],[]]
        self.additional_imgs_during_training = additional_imgs_during_training
        self.batch_size = batch_size,
        self.num_batches = num_batches

        val_dataset = datasets.ImageFolder(root="./images/shapes_dataset/test", transform=model_transforms["val"])
        self.val_dataset = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=3)

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

        mask_0 = (y == 0)
        mask_1 = (y == 1)

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

        self.log("test_loss_epoch", loss, on_step=False, on_epoch=True) # Logs average loss at the end of each epoch
        self.log("test_acc_epoch", acc, on_step=False, on_epoch=True) # Logs average loss at the end of each epoch
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def _get_batch_from_image(self, image, category, n=100):
        x = torch.stack([model_transforms["train"](image) for _ in range(n)])
        y = torch.full((n,), category, dtype=torch.long)
        return x, y
    
    def _merge_batches(self, batches):
        xs, ys = zip(*batches)  # unzip into two lists
        merged_x = torch.cat(xs, dim=0)
        merged_y = torch.cat(ys, dim=0)
        return merged_x, merged_y

    def online_training_step(self, batch):
        # batch = self._get_batch_from_image(image, category, n=200)
        optimizer = self.configure_optimizers()
        loss = self.training_step(batch, 0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _get_additional_images(self, category):
            
        other = 1 - category

        # if len(seen[other]) == 0:
        #     return None
        
        other_num_imgs = self.additional_imgs_during_training // 2 + 1
        same_num_imgs = self.additional_imgs_during_training // 2

        other_imgs = self.seen[other] if len(self.seen[other]) <= other_num_imgs else random.sample(self.seen[other], k=other_num_imgs)
        same_imgs = self.seen[category] if len(self.seen[category]) <= same_num_imgs else random.sample(self.seen[category],k=same_num_imgs)
        return other_imgs, same_imgs

    def train_online(self, image_b64, category):
        other = 1 - category
        additional_imgs = 3
        batch_size = 500
        num_batches = 1
        batch_size_per_image = batch_size // (additional_imgs+1)
        other_imgs, same_imgs = self._get_additional_images(category)
        for _ in range(num_batches):
            batches = [self._get_batch_from_image(Image.open(BytesIO(base64.b64decode(image))), other, n=batch_size_per_image) for image in other_imgs] + \
                [self._get_batch_from_image(Image.open(BytesIO(base64.b64decode(image))), category, n=batch_size_per_image) for image in same_imgs] + \
                [self._get_batch_from_image(Image.open(BytesIO(base64.b64decode(image_b64))), category, n=batch_size_per_image)]
            batch = self._merge_batches(batches)
            self.online_training_step(batch)
        # self.run_validation(self.val_dataset)
        self.seen[category].append(image_b64)
        self.val_step_counter += 1
        

class ModelHandler:

    def __init__(self):
        self.dense_model = DenseModel(image_size=20, hidden_layer1_size=200, output_size=2)
        # self.perfect_model = PerfectModel()
        self.model = self.dense_model
        print("Loaded all models.")

    def predict(self, image_b64):
        encoded_string = base64.b64decode(image_b64)
        image_data = Image.open(BytesIO(encoded_string))
        # image_data.save(f"./images/temp/{uuid.uuid1()}.png")
        with torch.no_grad():
            prediction = torch.softmax(self.model(model_transforms["val"](image_data))[0], dim=-1)
        label = torch.argmax(prediction).tolist()
        confidence = 200*(float(torch.max(prediction))-0.5)
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
            prediction = torch.softmax(self.model(model_transforms["val"](image_data))[0], dim=-1)
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

    def train(self, image_b64, category):
        self.model.train_online(image_b64, category)