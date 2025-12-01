import io
import random
import json
from api.api import app

import pytest
from fastapi.testclient import TestClient
from PIL import Image

client = TestClient(app)

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
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    response = client.post(
        "/predict",
        files={"data": ("hamis.jpg", img_bytes, "image/jpeg")},
    )

    assert json.loads(response.content)["result"] == expected


@pytest.mark.parametrize("width,height,expected",[
    (30, 30, [30, 30]),
    (500, 500, [500, 500]),
    (1000, 30, [1000, 30]),
])
def test_resize(def_image, width, height, expected):
    img = Image.open(def_image)
    img.convert("RGB")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    data = {"width": width, "height": height}
    response = client.post(
        "/resize",
        data=data,
        files={"data": ("hamis.jpg", img_bytes, "image/jpeg")},
    )
    assert json.loads(response.content)["result"] == expected
