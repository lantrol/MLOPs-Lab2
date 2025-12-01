import io
import random
import json
from mylib.predict import (
    predict_image,
    resize_image
)

import pytest
from PIL import Image


# -------- Fixtures ----------

@pytest.fixture
def def_image():
    return "./samples/hamis.jpg"


# -------- Tests ---------

@pytest.mark.parametrize("seed,expected",[
    (1, "Motorcicle"),
    (2, "Car"),
    (3, "Motorcicle"),
])
def test_predict(def_image, seed, expected):
    random.seed(seed)
    img = Image.open(def_image)
    img.convert("RGB")
    result = predict_image(img)

    assert result == expected


@pytest.mark.parametrize("width,height,expected",[
    (30, 30, (30, 30)),
    (500, 500, (500, 500)),
    (1000, 30, (1000, 30)),
])
def test_resize(def_image, width, height, expected):
    img = Image.open(def_image)
    img.convert("RGB")

    result = resize_image(img, width, height)

    assert result == expected
