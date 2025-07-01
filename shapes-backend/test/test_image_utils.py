from image_utils import CenterGreyscaleImage, get_batch_from_images
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image, ImageDraw
from PIL.ImageOps import invert
import numpy as np

def test_find_shape_center():
    center_tfm = CenterGreyscaleImage()
    img = Image.open("./images/shapes_dataset/test/circle/0c211962-3c29-11f0-8e09-00155ddb6ef7.png")
    img = img.convert("L")
    img = invert(img)
    cx, cy = center_tfm.calc_center(img)
    draw = ImageDraw.Draw(img)
    draw.circle([cx, cy], radius=20, fill=255)
    img.show()
    print(cx, cy)


def test_center_shape():
    center_tfm = CenterGreyscaleImage()
    img = Image.open("./images/shapes_dataset/test/circle/0c211962-3c29-11f0-8e09-00155ddb6ef7.png")
    img = img.convert("L")
    img = invert(img)
    img.show()
    img = center_tfm(img)
    img.show()

def test_transform_images():
    img = Image.open("./images/shapes_dataset/test/circle/0c211962-3c29-11f0-8e09-00155ddb6ef7.png")

    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomInvert(p=1.0),
        CenterGreyscaleImage(),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.25, 0.25),
            scale=(1, 1.5),
            shear=5,
            fill=0,
        ),                 # Converts to tensor (shape: [1, H, W])
        transforms.RandomInvert(p=1.0),
    ])

    for i in range(20):
        img_tfm = test_transforms(img)
        img_tfm.save(f"./images/temp/{i}.jpg")

def test_img_same_after_centering():
    img = Image.open("./images/shapes_dataset/test/circle/0c211962-3c29-11f0-8e09-00155ddb6ef7.png")

    tfm1 = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomInvert(p=1.0),
    ])

    tfm2 = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomInvert(p=1.0),
        CenterGreyscaleImage(),
    ])

    img_tfm1 = tfm1(img)
    img_tfm2 = tfm2(img)

    img_tfm1.show()
    img_tfm2.show()
    print(np.mean(np.array(img_tfm1)))
    print(np.mean(np.array(img_tfm2)))


def test_get_batch_from_images():
    images = [
        Image.open("images/shapes_dataset/test/circle/0c211962-3c29-11f0-8e09-00155ddb6ef7.png"),
        Image.open("images/shapes_dataset/test/circle/062656a8-3c29-11f0-8e09-00155ddb6ef7.png")
    ]
    x, y = get_batch_from_images(images, 0)
    print(x.shape)
    print(y.shape)


if __name__ == "__main__":
    test_get_batch_from_images()