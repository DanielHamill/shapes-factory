from PIL import Image
from io import BytesIO
import base64
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_handler import DenseModel, PerfectModel
import tqdm
import mlflow
import random


# TODO: convert these to a proper testing framework like pytest later

model = None
trainer = None
train_dataloader = None
val_dataloader = None

def load_model():
    global model
    global trainer
    global train_dataloader
    global val_dataloader
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
        transforms.ToTensor(),                        # Converts        to tensor (shape: [1, H, W])
        # transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize for 1 channel
    ])

    train_dataset = datasets.ImageFolder(root="./images/shapes_dataset/train", transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3)

    val_dataset = datasets.ImageFolder(root="./images/shapes_dataset/test", transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=3)

    model = DenseModel(image_size=20, hidden_layer1_size=200, output_size=2)

    mlf_logger = MLFlowLogger(
        experiment_name="lightning_logs",
        tracking_uri="file:./ml-runs",
        checkpoint_path_prefix="my_prefix"
    )
    trainer = L.Trainer(max_epochs=100, logger=mlf_logger)

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
    load_model()
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

def test_train_online():
    load_model()
    # x, y = model._get_batch_from_image(image, 0, n=32)

    dataset_root = "./images/shapes_dataset/train"
    circles = [f"{dataset_root}/circle/{filename}" for filename in os.listdir(f"{dataset_root}/circle")]
    triangles = [f"{dataset_root}/triangle/{filename}" for filename in os.listdir(f"{dataset_root}/triangle")]

    mlflow.set_tracking_uri("file:./ml-runs")  # if needed
    mlflow.set_experiment("my_experiment_name")  # optional

    # with mlflow.start_run(run_name="validate_multiple_times") as run:
    # You can log params here too if needed

    # i = 0
    # seen = [[],[]]

    # additional_imgs = 1

    # def get_additional_images(category):
            
    #     other = 1 - category

    #     # if len(seen[other]) == 0:
    #     #     return None
        
    #     other_num_imgs = additional_imgs // 2 + 1
    #     same_num_imgs = additional_imgs // 2

    #     other_imgs = seen[other] if len(seen[other]) <= other_num_imgs else random.sample(seen[other], k=other_num_imgs)
    #     same_imgs = seen[category] if len(seen[category]) <= same_num_imgs else random.sample(seen[category],k=same_num_imgs)
    #     return other_imgs, same_imgs


    for images in tqdm.tqdm(zip(circles, triangles)):
        for category, image in enumerate(images):
            # other = 1 - category
            # additional_imgs = 3
            # batch_size = 500
            # num_batches = 1
            # batch_size_per_image = batch_size // (additional_imgs+1)
            # other_imgs, same_imgs = get_additional_images(category)
            # for b in range(num_batches):
            #     batches = [model._get_batch_from_image(Image.open(image), other, n=batch_size_per_image) for image in other_imgs] + \
            #         [model._get_batch_from_image(Image.open(image), category, n=batch_size_per_image) for image in same_imgs] + \
            #         [model._get_batch_from_image(Image.open(image), category, n=batch_size_per_image)]
            #     batch = model._merge_batches(batches)

            #     model.custom_step = i
            #     model.online_train(batch)
            # model.run_validation(val_dataloader)
            # seen[category].append(image)
            # i += 1
            with open(image, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            model.train_online(encoded_string, category)

if __name__ == "__main__":
    test_train_online()
    # test_train_model()