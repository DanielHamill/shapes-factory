from image_utils import CenterGreyscaleImage
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image, ImageDraw
from PIL.ImageOps import invert

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

if __name__ == "__main__":
    test_center_shape()