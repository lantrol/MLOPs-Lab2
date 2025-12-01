# Import the libraries, classes and functions
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from mylib.predict import predict_image, resize_image
from PIL import Image
import io

# Create an instance of FastAPI
app = FastAPI(
    title="API for image prediction using FastAPI",
    description="API to perform image prediction",
    version="1.0.0",
)

# We use the templates folder to obtain HTML files
templates = Jinja2Templates(directory="templates")


# Initial endpoint
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")


# Main endpoint to perform the artihmetical operations using the input class defined with Pydantic
@app.post("/predict")
async def predict(data: UploadFile = File(...)):
    """
    It performs an arithmetical operation according to the input parameters.
    """
    # op = data.operation.lower()
    file = await data.read()
    image = Image.open(io.BytesIO(file))

    result = predict_image(image)
    return {"result": result}


@app.post("/resize")
async def resize(
    width: int = Form(), height: int = Form(), data: UploadFile = File(...)
):
    """
    It performs an arithmetical operation according to the input parameters.
    """

    if width <= 0 or height <= 0:
        return HTTPException(500, "Invalid width or height")

    file = await data.read()
    image = Image.open(io.BytesIO(file))

    result = resize_image(image, width, height)
    return {"result": result}


# Entry point (for direct execution only)
if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
