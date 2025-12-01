import json
import ast
import math
import random

from mylib.predict import (
    predict_image,
    resize_image
)
from cli.cli import cli

import pytest
from click.testing import CliRunner


# ------- Fixtures --------

@pytest.fixture
def def_image():
    return "./samples/hamis.jpg"


# -------- Tests ----------

@pytest.mark.parametrize("seed,expected",[
    (1, "Motorcicle"),
    (2, "Car"),
    (3, "Motorcicle"),
])
def test_predict(def_image, seed, expected):
    runner = CliRunner()

    random.seed(seed)
    result = runner.invoke(cli, ["predict", def_image])
    assert result.output.strip() == expected


@pytest.mark.parametrize("width,height,expected",[
    (30, 30, "(30, 30)"),
    (500, 500, "(500, 500)"),
    (1000, 30, "(1000, 30)"),
])
def test_resize(def_image, width, height, expected):
    runner = CliRunner()
    result = runner.invoke(cli, ["resize", def_image, "--width", str(width), "--height", str(height)])
    assert result.output.strip() == expected
