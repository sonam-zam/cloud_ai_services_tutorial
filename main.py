from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torchvision.models as models
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

app = FastAPI(    title="Image Prediction",
    description="Classify and predict image",
    version="1.0.0",
    docs_url="/swagger",
    redoc_url=None,)



model = models.resnet18(pretrained=True)
model.eval()
LABELS = open("imagenet_classes.txt").read().strip().split("\n")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


PREDICTION_KEY = "00e15d939583442eaedae66af2523e92"
ENDPOINT_URL = "https://eastus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/6367cdde-d0f8-4cf9-b867-1b1793ad7e76/classify/iterations/Iteration1/image"

def classify_image(image_data):
    headers = {
        "Prediction-Key": "00e15d939583442eaedae66af2523e92",
    "Content-Type": "application/json",
    }
    response = requests.post(ENDPOINT_URL, headers=headers, data=image_data)
    return response.json()


@app.post("/classify/")
async def upload_file(file: UploadFile = File(...)):
    image_data = await file.read()
    result = classify_image(image_data)
    return result


@app.get("/openapi.json")
async def get_openapi():
        return JSONResponse(content=get_openapi(title="Image Prediction", version="1.0.0"))


# Ensure that OPTIONS requests are allowed for this resource
@app.options("/openapi.json")
async def options_openapi():
    return {"detail": "Allow OPTIONS for OpenAPI schema"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML service"}

@app.get("/model_name")
def get_model_name():
    return {"model_name": "ResNet18"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_stream = io.BytesIO(await file.read())
    image = Image.open(image_stream).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    _, predicted_idx = torch.max(output, 1)
    predicted_label = LABELS[predicted_idx.item()]

    return JSONResponse(content={"predicted_label": predicted_label})
