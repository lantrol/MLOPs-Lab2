from PIL import Image
import random

from .model import ImageModel


def predict_image(image):
    model = ImageModel()
    return model.predict(image)


def resize_image(image, width: int, height: int):
    image = image.resize((width, height))
    return image.size


if __name__ == "__main__":
    img = Image.open("samples/hamis.jpg")
    img = resize_image(img, 32, 32)

    random.seed(1)
    print(predict_image(img))
    random.seed(2)
    print(predict_image(img))
    random.seed(3)
    print(predict_image(img))
