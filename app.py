import io

import gradio as gr
import requests as rq
from PIL import Image

API_URL = "https://mlops-lab-latest.onrender.com"


def predict(img):
    try:
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format="png")
        imgByteArr = imgByteArr.getvalue()
        payload = {"data": imgByteArr}
        response = rq.post(f"{API_URL}/predict", files=payload, timeout=None)
        response.raise_for_status()
        data = response.json()
        return data.get("result")
    except rq.exceptions.HTTPError as e:
        return f"Error: {response.json().get('detail', str(e))}"


with gr.Blocks() as demo:
    gr.Markdown("# Page")
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil")
            btn = gr.Button("Predict")
        cls = gr.Textbox()

    btn.click(predict, inputs=img, outputs=cls)

demo.launch()
