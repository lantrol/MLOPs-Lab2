from mylib.predict import predict_image, resize_image

import click
from PIL import Image


# ===== GENERAL CLI GROUP =====
@click.group()
def cli():
    """CLI tool for data preprocessing and transformation tasks."""


@cli.command("predict")
@click.argument("img_path", type=str)
def predict(img_path):
    img = Image.open(img_path)
    result = predict_image(img)
    click.echo(result)


@cli.command("resize")
@click.argument("img_path", type=str)
@click.option("--width", "width", type=int, default=64)
@click.option("--height", "height", type=int, default=64)
def resize(img_path, width, height):
    img = Image.open(img_path)
    result = resize_image(img, width, height)
    click.echo(result)


# ===== MAIN ENTRY =====
if __name__ == "__main__":
    cli()
